import os
import gc
import time
from tqdm import tqdm

import cv2
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from core.loss import CharbonnierLoss
from core.dataset import TestDataset
from core.metrics import calc_psnr_and_ssim
from core.utils import get_ref_index

from model.modules.flow_comp_raft import RAFT_bi
from model.modules.depth_anything_v2.dpt import DepthAnythingV2


class Trainer:
    def __init__(self, config, prefetcher, model, start_epoch=0):

        self.l1_loss = nn.L1Loss()
        self.charbonnier_loss = CharbonnierLoss()

        self.config = config
        self.device = config["device"]

        # ---- distributed setup ----
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main = self.rank == 0

        # self.net is always the bare module (checkpoints, img_propagation,
        # validation); self.model is the (possibly DDP/compiled) training entry.
        self.net = model
        if self.world_size > 1:
            self.model = DDP(
                model,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=config["trainer"].get("ddp_find_unused", True),
            )
        else:
            self.model = model
        if config["trainer"].get("compile", False):
            self.model = torch.compile(self.model)
        self.epoch = start_epoch
        self.iteration = 0
        self.num_local_frames = config["dl_config"]["num_local_frames"]
        self.num_ref_frames = config["dl_config"]["num_ref_frames"]

        self.train_args = config["trainer"]

        # Mixed-precision (AMP) settings. Defaults match the released training
        # run (bf16 autocast + GradScaler). Set trainer.amp=false for full fp32,
        # or trainer.amp_dtype="float16" for fp16.
        amp_dtypes = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        self.use_amp = self.train_args.get("amp", True)
        self.amp_dtype = amp_dtypes[self.train_args.get("amp_dtype", "bfloat16")]
        # GradScaler is only meaningful for fp16; bf16 has fp32 range so it uses
        # a plain backward/step. Stabilisers: gradient clipping + NaN/Inf guard.
        self.use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.grad_clip = self.train_args.get("grad_clip", 1.0)
        self.nan_skips = 0
        self.best_psnr = float("-inf")  # best validation PSNR (see validate())

        self.prefetcher = prefetcher

        # Initialize RAFT
        self.raft = RAFT_bi(device=self.device)

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        encoder = "vits"  # or 'vitb', 'vitl', 'vitg'

        depth_model = DepthAnythingV2(**model_configs[encoder])
        depth_model.load_state_dict(
            torch.load(
                f"./weights/depth_anything_v2_{encoder}.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        self.depth_model = depth_model.to(self.device).eval()

        self.interp_mode = self.config["interp_mode"]

        # Predictions are 2x the input resolution; built once here instead of
        # per iteration (the train crop size is fixed).
        self.resize_to_input = transforms.Resize(
            size=(
                config["dl_config"]["h_train"] // 2,
                config["dl_config"]["w_train"] // 2,
            ),
            antialias=None,
        )

        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.scaler = torch.GradScaler("cuda", enabled=self.use_scaler)
        self.load()

        # ---- EMA of the model weights (rank 0 only; ranks stay identical).
        # Validation and best-model checkpoints use the EMA weights.
        self.ema_decay = self.train_args.get("ema_decay", 0.999)
        self.ema = None
        if self.is_main:
            ema_path = os.path.join(config["out_dir"], "model_ema_latest.pth")
            if os.path.isfile(ema_path):
                self.ema = torch.load(ema_path, map_location="cuda", weights_only=True)
                print("Resumed EMA weights from model_ema_latest.pth")
            else:
                self.ema = {
                    k: v.detach().clone().float()
                    for k, v in self.net.state_dict().items()
                }

        # Set up tensorboard (rank 0 only)
        self.summary = {}
        self.log_dir = os.path.join(config["out_dir"], "logs")
        self.writer = SummaryWriter(self.log_dir) if self.is_main else None

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
            else:
                print(f"Params {name} will not be optimized.")

        optim_params = [
            {"params": backbone_params, "lr": self.config["trainer"]["lr"]},
        ]

        tcfg = self.config["trainer"]
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=tcfg["lr"],
            betas=tuple(tcfg.get("betas", (0.9, 0.999))),
            eps=tcfg.get("eps", 1e-8),
            weight_decay=tcfg.get("weight_decay", 0.01),
        )

    def setup_schedulers(self):
        """Per-iteration LR schedule: optional linear warmup then the configured
        decay (honours config["trainer"]["scheduler"], default CosineAnnealingLR).
        Stepped once per optimiser step in _train_epoch (not per epoch)."""
        tcfg = self.config["trainer"]
        total_iters = int(tcfg["iterations"])
        warmup_iters = int(tcfg.get("warmup_iters", 0))
        sched_cfg = tcfg.get("scheduler", {}) or {}
        sched_type = sched_cfg.get("type", "CosineAnnealingLR")

        if sched_type == "CosineAnnealingLR":
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_iters - warmup_iters),
                eta_min=sched_cfg.get("eta_min", 0.0),
            )
        elif sched_type == "StepLR":
            main = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get("step_size", 100000),
                gamma=sched_cfg.get("gamma", 0.1),
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

        if warmup_iters > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=sched_cfg.get("warmup_start_factor", 0.01),
                end_factor=1.0,
                total_iters=warmup_iters,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup, main], milestones=[warmup_iters]
            )
        else:
            self.scheduler = main

    def add_summary(self, writer, name, val):
        """
        Add tensorboard summary.

        """
        if writer is None:  # non-main ranks don't log
            return
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        writer.add_scalar(name, self.summary[name], self.iteration)
        self.summary[name] = 0

    def load(self):
        """Resume from the newest checkpoint in out_dir, if any.

        Prefers the rolling ``model_latest.pth``; falls back to legacy
        numbered checkpoints (``model_NNNNNN.pth`` via ``latest.ckpt``), then
        to ``trainer.model_path`` in the config (pretrained init). The LR
        schedule is always rebuilt from the *current* config and replayed to
        the resumed iteration — identical to an uninterrupted run for the
        same config, and it lets a resume extend or shorten the training
        horizon (the cosine then anneals to ``eta_min`` at the new total).
        """
        out_dir = self.config["out_dir"]

        if os.path.isfile(os.path.join(out_dir, "model_latest.pth")):
            model_path = os.path.join(out_dir, "model_latest.pth")
            opt_path = os.path.join(out_dir, "opt_latest.pth")
        elif os.path.isfile(os.path.join(out_dir, "latest.ckpt")):
            latest_it = (
                open(os.path.join(out_dir, "latest.ckpt"), "r").read().splitlines()[-1]
            )
            model_path = os.path.join(out_dir, f"model_{int(latest_it):06d}.pth")
            opt_path = os.path.join(out_dir, f"opt_{int(latest_it):06d}.pth")
        else:
            model_path = self.train_args.get("model_path", None)
            opt_path = self.train_args.get("opt_path", None)

        if model_path is None:
            print(
                "Warning: There is no trained model found by trainer.py.\n"
                "A randomly initialized model will be used."
            )
            return

        print(f"Loading model from {model_path}")
        model_data = torch.load(model_path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(model_data)

        if opt_path is not None and os.path.isfile(opt_path):
            data_opt = torch.load(opt_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(data_opt["optimizer"])
            if "scaler" in data_opt:
                self.scaler.load_state_dict(data_opt["scaler"])
            self.epoch = data_opt["epoch"]
            self.iteration = data_opt["iteration"]

            # Replay the config-built schedule to the resumed iteration (the
            # saved scheduler state is ignored on purpose, see docstring).
            for _ in range(int(self.iteration)):
                self.scheduler.step()
            print(
                f"Resumed at iteration {self.iteration} "
                f"(lr {self.scheduler.get_last_lr()[0]:.3e})"
            )

        best_file = os.path.join(out_dir, "best.ckpt")
        if os.path.isfile(best_file):
            best_it, best_psnr = open(best_file, "r").read().split()
            self.best_psnr = float(best_psnr)
            print(f"Best validation PSNR so far: {self.best_psnr:.2f} (iter {best_it})")

    @staticmethod
    def _atomic_save(obj, path):
        """torch.save via a temp file + rename, so an interrupted write (e.g.
        a crash or reboot mid-save) can never corrupt the previous checkpoint."""
        tmp_path = path + ".tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    def save(self, it):
        """Save the rolling 'latest' checkpoint (overwritten each time, so a
        run keeps one resumable checkpoint plus the best-validation model
        saved by validate() — not one pair per save_freq)."""
        model_path = os.path.join(self.config["out_dir"], "model_latest.pth")
        opt_path = os.path.join(self.config["out_dir"], "opt_latest.pth")
        print(f"\nsaving model to {model_path} (iteration {it})")

        self._atomic_save(self.net.state_dict(), model_path)
        if self.ema is not None:
            self._atomic_save(
                self.ema, os.path.join(self.config["out_dir"], "model_ema_latest.pth")
            )
        self._atomic_save(
            {
                "epoch": self.epoch,
                "iteration": self.iteration,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            opt_path,
        )

        # Informational record of the last saved iteration.
        with open(os.path.join(self.config["out_dir"], "latest.ckpt"), "w") as f:
            f.write(f"{it:06d}\n")

    @torch.no_grad()
    def _ema_update(self):
        """Exponential moving average of the weights, with the usual ramp-up
        so early steps don't anchor the average to the random init."""
        d = min(self.ema_decay, (1 + self.iteration) / (10 + self.iteration))
        for k, v in self.net.state_dict().items():
            e = self.ema[k]
            if v.dtype.is_floating_point:
                e.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                e.copy_(v)

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args["iterations"]))
        pbar = tqdm(
            pbar,
            initial=self.iteration,
            dynamic_ncols=True,
            smoothing=0.01,
            disable=not self.is_main,
        )

        while True:
            self.epoch += 1
            pbar.set_postfix(epoch=self.epoch)
            self.prefetcher.reset()
            torch.cuda.empty_cache()
            gc.collect()
            # Validation is now periodic (every val_freq iters) inside
            # _train_epoch; the LR scheduler steps per-iteration there too.
            self._train_epoch(pbar)
            if self.iteration > self.train_args["iterations"]:
                break
        print("\nTraining complete.")

    def _train_epoch(self, pbar):
        """
        Process input and calculate loss every training epoch
        """

        device = self.device
        self.model.train()  # restore train mode (validate() switches to eval)
        train_data = self.prefetcher.next()
        while train_data is not None:
            # Workers build batches on CPU in parallel; move this one to the GPU.
            train_data = [
                d.to(device, non_blocking=True) if torch.is_tensor(d) else d
                for d in train_data
            ]
            with torch.autocast(device_type=device, dtype=self.amp_dtype, enabled=self.use_amp):
                self.iteration += 1
                gt_frames, resized_gt_frames, blurry_frames, blur_maps, local_index = train_data
                l_t = self.num_local_frames
                b, t, c, h, w = blurry_frames.size()

                # Create Depth maps of blurred images
                depths = []
                with torch.no_grad():
                    for batch in blurry_frames:
                        depth_batch = self.depth_model.preprocess_tensor(batch, input_size=(294, 518))
                        depth_batch = self.depth_model.forward(depth_batch)
                        for depth in depth_batch:
                            depth = nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), (h, w), mode="bilinear", align_corners=True)[0, 0]
                            depth = 1 - ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8))
                            depths.append(depth.unsqueeze(0))
                    depths = torch.stack(depths).to(device).view(b, t, 1, h, w)

                # Create Binary Masks
                binary_masks = (blur_maps > torch.min(blur_maps)).float()

                # Get GT Optical Flow
                blurry_flows_bi = self.raft(blurry_frames)


                # ---- Image Propagation ----
                prop_frames, prop_masks = self.net.img_propagation(
                    blurry_frames,
                    blurry_flows_bi,
                    binary_masks,
                    interpolation=self.interp_mode,
                )

                # Dilate the binary masks
                prop_masks = torch.stack(
                    [nn.functional.max_pool2d(batch, kernel_size=3, stride=1, padding=1) for batch in prop_masks]
                )
                
                # ---- Transformer + Super Resolution ----
                ori_pred_frames = self.model(
                    prop_frames,
                    depths,
                    blurry_flows_bi,
                    blur_maps,
                    prop_masks,
                    l_t,
                )


                pred_frames = torch.stack(
                    [self.resize_to_input(batch) for batch in ori_pred_frames]
                )
                pred_frames = pred_frames.view(b, -1, c, h, w)


                # ---- Loss Calculation ----

                # Student l1 loss
                blur_loss = self.l1_loss(
                    pred_frames * binary_masks, resized_gt_frames * binary_masks
                )
                blur_loss = (
                    blur_loss
                    / torch.mean(binary_masks)
                    * self.config["losses"]["blur_weight"]
                )
                clear_loss = self.l1_loss(
                    pred_frames * (1 - binary_masks), resized_gt_frames * (1 - binary_masks)
                )
                clear_loss = (
                    clear_loss
                    / torch.mean(1 - binary_masks)
                    * self.config["losses"]["clear_weight"]
                )
                # Super Resolution Loss
                if self.config["losses"]["sr_weight"] > 0:
                    sr_loss = (
                        self.charbonnier_loss(ori_pred_frames, gt_frames)
                        * self.config["losses"]["sr_weight"]
                    )
                else:
                    sr_loss = 0

                total_loss = sr_loss + blur_loss + clear_loss
                # total_loss = self.l1_loss(ori_pred_frames, gt_frames)

                # Log only every log_freq iters; each .item() is a GPU->CPU sync.
                if self.iteration % self.train_args["log_freq"] == 0:
                    self.add_summary(self.writer, "loss/blur_loss", blur_loss.item())
                    self.add_summary(self.writer, "loss/clear_loss", clear_loss.item())
                    if self.config["losses"]["sr_weight"] > 0:
                        self.add_summary(self.writer, "loss/sr_loss", sr_loss.item())
                    self.add_summary(self.writer, "loss/total_loss", total_loss.item())
                    self.add_summary(
                        self.writer, "loss/learning_rate", self.scheduler.get_last_lr()[0]
                    )
                
            # ---- NaN/Inf-guarded, grad-clipped optimisation step ----
            # Under DDP the skip decision must be collective: if one rank
            # backwards while another skips, the gradient allreduce deadlocks.
            finite = torch.isfinite(total_loss.detach())
            if self.world_size > 1:
                flag = finite.int()
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                finite = flag.bool()
            if finite:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)  # no-op when scaler disabled
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                if torch.isfinite(grad_norm):
                    self.scaler.step(self.optimizer)
                    self.scheduler.step()
                    if self.ema is not None:
                        self._ema_update()
                else:
                    self.nan_skips += 1  # non-finite grads: skip this step
                self.scaler.update()
                if self.iteration % self.train_args["log_freq"] == 0:
                    self.add_summary(self.writer, "train/grad_norm", float(grad_norm))
                    self.add_summary(self.writer, "train/nan_skips", self.nan_skips)
            else:
                self.nan_skips += 1  # non-finite loss: skip backward entirely
            self.optimizer.zero_grad(set_to_none=True)

            ####################################################
            # Write images to tensorboard
            ####################################################
            if self.writer is not None and self.iteration % self.train_args["img_freq"] == 0:
                # tensors to cpu

                blur = torch.cat([i for i in blurry_frames[0].cpu()], 1)
                prop = torch.cat([i for i in prop_frames[0].cpu()], 1)
                output = torch.cat([i for i in pred_frames[0].cpu()], 1)
                gt = torch.cat([i for i in resized_gt_frames[0].cpu()], 1)

                img_results = torch.cat([blur, prop, output, gt], 2)
                img_results = torchvision.utils.make_grid(
                    img_results, nrow=1, normalize=True
                )
                self.writer.add_image(
                    f"img/gt", img_results, self.iteration
                )

                # depth, blur, binary mask and updated binary mask to cpu
                maps = torch.cat([i for i in blur_maps[0].cpu()], 1)
                masks = torch.cat([i for i in binary_masks[0].cpu()], 1)
                updated_masks = torch.cat([i for i in prop_masks[0].cpu()], 1)
                depth = torch.cat([i for i in depths[0].cpu()], 1)

                binary_results = torch.cat([maps, masks, updated_masks, depth], 2)
                self.writer.add_image(
                    "img/bin:blur-mask-up_mask-depth",
                    binary_results,
                    self.iteration,
                )

            # console logs
            pbar.update(1)
            if self.iteration % self.train_args["log_freq"] == 0:
                pbar.set_description(
                    (f"LR: {self.scheduler.get_last_lr()[0]} Loss: {total_loss.item():.3f}")
                )

            # saving models (rank 0 owns all checkpoints)
            if self.is_main and self.iteration % self.train_args["save_freq"] == 0:
                self.save(int(self.iteration))

            # periodic validation (validate() restores train mode on exit).
            # Rank 0 only; other ranks simply proceed and block on the next
            # gradient allreduce until rank 0 rejoins.
            # Guarded so a validation error can never kill an unattended run.
            val_freq = self.train_args.get("val_freq", 0)
            if self.is_main and val_freq and self.iteration % val_freq == 0:
                try:
                    self.validate()
                except Exception as e:
                    print(f"\n[warn] validation failed at iter {self.iteration}: {e}")
                    self.model.train()

            if self.iteration > self.train_args["iterations"]:
                break

            train_data = self.prefetcher.next()


    def validate(self):
        """Validate the EMA weights (falling back to the raw weights when no
        EMA is kept) and restore the training weights afterwards, even if
        validation throws."""
        if self.ema is None:
            self._validate_impl()
            return
        backup = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(
            {k: v.to(backup[k].dtype) for k, v in self.ema.items()}
        )
        try:
            self._validate_impl()
        finally:
            self.net.load_state_dict(backup)
            self.model.train()

    def _validate_impl(self):
        """Validate on a small fixed subset of DAVIS-Blur (first
        `val_num_videos` sequences; 0 = all 90). Mirrors the inference loop of
        test_dabit.py: RAFT flows, image propagation, per-frame cached depth,
        sliding transformer windows. Logs PSNR/SSIM to TensorBoard and to
        validation.txt in the run's output directory."""

        results_dict = {"PSNR": [], "SSIM": [], "Time": []}
        logfile = open(os.path.join(self.config["out_dir"], "validation.txt"), "a")
        logfile.write(f"Iteration: {self.iteration}")

        print("Validating model:")

        val_num = self.train_args.get("val_num_videos", 0)
        test_ds = TestDataset("./blur_tests")
        valid_loader = DataLoader(
            dataset=test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()

        pbar = tqdm(total=val_num or len(test_ds), leave=False)

        for vi, (gt_tensors, input_tensors, blur_maps, video_name) in enumerate(
            valid_loader
        ):
            if val_num and vi >= val_num:
                break
            video_start_time = time.time()

            # TestDataset returns CPU tensors (so the loader can prefetch); move
            # them onto the device for the GPU pipeline.
            gt_tensors = gt_tensors.to(self.device, non_blocking=True)
            input_tensors = input_tensors.to(self.device, non_blocking=True)
            blur_maps = blur_maps.to(self.device, non_blocking=True)

            h, w = input_tensors[0].shape[-2:]
            out_h, out_w = h * 2, w * 2

            binary_masks = torch.ones_like(blur_maps)

            video_length = input_tensors.shape[1]
            pbar.set_description(f"Processing: {video_name[0]} ({video_length} frames)")

            # ---- RAFT flows (fp32) ----
            short_clip_len = 12
            if video_length > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    s_f = f if f == 0 else f - 1
                    with torch.no_grad():
                        flows_f, flows_b = self.raft(
                            input_tensors[:, s_f:end_f], iters=20
                        )
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                flows_bi = (
                    torch.cat(gt_flows_f_list, dim=1),
                    torch.cat(gt_flows_b_list, dim=1),
                )
            else:
                with torch.no_grad():
                    flows_bi = self.raft(input_tensors, iters=20)
            torch.cuda.empty_cache()

            # ---- image propagation ----
            subvideo_length_img_prop = 100

            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = binary_masks[:, s_f:e_f].size()
                    pred_flows_bi_sub = (
                        flows_bi[0][:, s_f : e_f - 1],
                        flows_bi[1][:, s_f : e_f - 1],
                    )
                    prop_imgs_sub, updated_local_masks_sub = (
                        self.net.img_propagation(
                            input_tensors[:, s_f:e_f],
                            pred_flows_bi_sub,
                            binary_masks[:, s_f:e_f],
                            "nearest",
                        )
                    )
                    updated_frames_sub = (
                        input_tensors[:, s_f:e_f] * (1 - binary_masks[:, s_f:e_f])
                        + prop_imgs_sub.view(b, t, 3, h, w) * binary_masks[:, s_f:e_f]
                    )
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                    updated_frames.append(
                        updated_frames_sub[:, pad_len_s : e_f - s_f - pad_len_e]
                    )
                    updated_masks.append(
                        updated_masks_sub[:, pad_len_s : e_f - s_f - pad_len_e]
                    )
                    torch.cuda.empty_cache()

                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = binary_masks.size()
                prop_imgs, updated_local_masks = self.net.img_propagation(
                    input_tensors, flows_bi, binary_masks, interpolation="nearest"
                )
                updated_frames = (
                    input_tensors * (1 - binary_masks)
                    + prop_imgs.view(b, t, 3, h, w) * binary_masks
                )
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()

            # ---- depth prediction, cached once per frame (identical values to
            # recomputing inside the window loop, since the per-frame min-max
            # normalisation is independent of the window) ----
            depth_bs = 12
            frame_depths = [None] * video_length
            with torch.no_grad():
                for s in range(0, video_length, depth_bs):
                    e = min(video_length, s + depth_bs)
                    depth_batch = self.depth_model.preprocess_tensor(
                        updated_frames[0, s:e]
                    )
                    depth_batch = self.depth_model.forward(depth_batch).unsqueeze(1)
                    depth_batch = nn.functional.interpolate(
                        depth_batch, (h, w), mode="bilinear", align_corners=True
                    )
                    for k in range(depth_batch.shape[0]):
                        depth = depth_batch[k]
                        depth = 1 - (
                            (depth - depth.min()) / (depth.max() - depth.min())
                        )
                        frame_depths[s + k] = depth
            frame_depths = torch.stack(frame_depths)  # [video_length, 1, h, w]
            torch.cuda.empty_cache()

            # ---- sliding-window transformer inference ----
            comp_frames = [None] * video_length
            neighbor_stride = 5

            neighbor_ids = None
            for f in range(0, video_length, neighbor_stride)[:-1]:
                if f + 10 > video_length:
                    start, end = video_length - 10, video_length
                else:
                    start, end = f, f + 10
                neighbor_ids = list(range(start, end))
                ref_ids = get_ref_index(video_length, 6, neighbor_ids)

                selected_imgs = updated_frames[:, neighbor_ids + ref_ids]
                selected_blur_maps = blur_maps[:, neighbor_ids + ref_ids]
                selected_update_masks = updated_masks[:, neighbor_ids + ref_ids]
                selected_pred_flows_bi = self.raft(selected_imgs, iters=20)
                selected_depths = frame_depths[neighbor_ids + ref_ids].unsqueeze(0)

                l_t = len(neighbor_ids)

                with torch.no_grad():
                    ori_pred_img = self.net(
                        selected_imgs,
                        selected_depths,
                        selected_pred_flows_bi,
                        selected_blur_maps,
                        selected_update_masks,
                        l_t,
                    )
                pred_img = ori_pred_img.view(-1, 3, out_h, out_w).permute(0, 2, 3, 1)
                pred_img = pred_img.clamp(0, 1)

                for i, idx in enumerate(neighbor_ids):
                    img = np.array(pred_img[i].cpu().numpy() * 255).astype(np.uint8)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (
                            comp_frames[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)

            torch.cuda.empty_cache()
            avg_runtime = (time.time() - video_start_time) / video_length
            results_dict["Time"].append(avg_runtime)

            # ---- metrics ----
            pbar.set_description(f"Computing metrics for {video_name[0]}")
            gt_frames = gt_tensors[0].permute(0, 2, 3, 1).cpu().numpy()

            psnr_list, ssim_list = [], []
            for comp_frame, gt_frame in zip(comp_frames, gt_frames):
                comp_frame = cv2.resize(
                    cv2.cvtColor(comp_frame, cv2.COLOR_BGR2RGB),
                    (gt_frame.shape[1], gt_frame.shape[0]),
                ).astype(np.float32)
                gt_frame = (gt_frame * 255).astype(np.float32)

                psnr, ssim = calc_psnr_and_ssim(comp_frame, gt_frame)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

            results_dict["PSNR"].append(np.mean(psnr_list))
            results_dict["SSIM"].append(np.mean(ssim_list))

            summary = (
                f"PSNR: {np.mean(psnr_list):.2f}, SSIM: {np.mean(ssim_list):.4f}, "
                f"Time: {avg_runtime:.4f}"
            )
            pbar.update(1)
            pbar.write(f"{video_name[0]} {summary}")
            logfile.write("\n" + "{:<15s} -- {}".format(f"{video_name[0]}", summary))

        msg = (
            "\n"
            + "{:<15s} -- {}".format(
                "Average",
                {k: round(np.mean(v), 4) for k, v in results_dict.items()},
            )
            + "\n \n"
        )

        avg_psnr = float(np.mean(results_dict["PSNR"]))
        self.writer.add_scalar("validation/avg_psnr", avg_psnr, self.iteration)
        self.writer.add_scalar(
            "validation/avg_ssim", np.mean(results_dict["SSIM"]), self.iteration
        )

        # Keep the best-validation model alongside the rolling latest one.
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            # self.net currently holds the EMA weights (see validate()), so the
            # best checkpoint is the EMA model.
            self._atomic_save(
                self.net.state_dict(),
                os.path.join(self.config["out_dir"], "model_best.pth"),
            )
            with open(os.path.join(self.config["out_dir"], "best.ckpt"), "w") as f:
                f.write(f"{self.iteration} {avg_psnr:.4f}\n")
            print(
                f"\nNew best validation PSNR {avg_psnr:.2f} "
                f"at iteration {self.iteration} -- saved model_best.pth"
            )

        print(msg, end="")
        logfile.write(msg)
        logfile.close()
        pbar.close()
        self.model.train()  # restore train mode for continued training
