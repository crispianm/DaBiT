import os
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

from core.loss import AdversarialLoss, PerceptualLoss, LPIPSLoss, CharbonnierLoss
from core.dataset import *

from model.modules.flow_comp_raft import RAFT_bi
from model.modules.depth_anything_v2.dpt import DepthAnythingV2


from core.metrics import calc_psnr_and_ssim
from RAFT.utils.flow_viz_pt import flow_to_image


class Trainer:
    def __init__(self, config, prefetcher, model, start_epoch=0):

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.charbonnier_loss = CharbonnierLoss()

        self.config = config
        self.model = model
        self.device = config["device"]
        self.epoch = start_epoch
        self.iteration = 0
        self.num_local_frames = config["dl_config"]["num_local_frames"]
        self.num_ref_frames = config["dl_config"]["num_ref_frames"]

        self.train_args = config["trainer"]

        self.prefetcher = prefetcher

        # Initialize RAFT
        self.raft = RAFT_bi(device=self.device)
        # self.fix_flow_complete = RecurrentFlowCompleteNet(
        #     "C:/Users/wg19671/repos/DepthPainter/experiments_model/recurrent_flow_completion_train_flowcomp/gen_108000.pth"
        # )
        # for p in self.fix_flow_complete.parameters():
        #     p.requires_grad = False
        # self.fix_flow_complete.to(self.device)
        # self.fix_flow_complete.eval()

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
        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        # self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)
        self.scaler = torch.GradScaler('cuda', enabled=True)
        self.load()

        # Set up tensorboard
        self.summary = {}
        self.log_dir = os.path.join(config["out_dir"], "logs")
        self.writer = SummaryWriter(self.log_dir)

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

        self.optimizer = torch.optim.AdamW(
            optim_params,
        )

    def setup_schedulers(self):
        """Set up schedulers."""
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size = 50,
            gamma = 0.5
        )
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode="max",
        #     factor=0.5,
        #     patience=5,
        #     threshold=0.01,  # metric to be used is psnr
        #     threshold_mode="abs",
        # )

    def add_summary(self, writer, name, val):
        """
        Add tensorboard summary.

        """
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        n = self.train_args["log_freq"]
        if writer is not None and self.iteration % n == 0:
            writer.add_scalar(name, self.summary[name] / n, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load DepthPainter"""
        # get the latest checkpoint
        # TODO: add resume name
        if os.path.isfile(os.path.join(self.config["out_dir"], "latest.ckpt")):
            latest_epoch = (
                open(os.path.join(self.config["out_dir"], "latest.ckpt"), "r")
                .read()
                .splitlines()[-1]
            )
        else:
            ckpts = [
                os.path.basename(i).split(".pth")[0]
                for i in glob.glob(os.path.join(self.config["out_dir"], "*.pth"))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1][4:] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            model_path = os.path.join(
                self.config["out_dir"], f"model_{int(latest_epoch):06d}.pth"
            )
            opt_path = os.path.join(
                self.config["out_dir"], f"opt_{int(latest_epoch):06d}.pth"
            )

            print(f"Loading model from {model_path}")
            model_data = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(model_data)

            data_opt = torch.load(opt_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(data_opt["optimizer"])
            self.scaler.load_state_dict(data_opt["scaler"])

            self.epoch = data_opt["epoch"]
            self.iteration = data_opt["iteration"]
        else:
            model_path = self.config["trainer"].get("model_path", None)
            opt_path = self.config["trainer"].get("opt_path", None)
            if model_path is not None:
                print(f"Loading Gen-Net from {model_path}")
                model_data = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(model_data)

                if opt_path is not None:
                    data_opt = torch.load(opt_path, map_location=self.device, weights_only=True)
                    self.optimizer.load_state_dict(data_opt["optimizer"])
                    self.scheduler.load_state_dict(data_opt["scheduler"])

            else:
                print(
                    "Warning: There is no trained model found by trainer.py.\n"
                    "A randomly initialized model will be used."
                )

    def save(self, it):
        """Save parameters every eval_epoch"""
        # configure path
        model_path = os.path.join(self.config["out_dir"], f"model_{it:06d}.pth")
        opt_path = os.path.join(self.config["out_dir"], f"opt_{it:06d}.pth")
        print(f"\nsaving model to {model_path} ")

        # remove .module for saving

        # save checkpoints
        torch.save(self.model.state_dict(), model_path)
        torch.save(
            {
                "epoch": self.epoch,
                "iteration": self.iteration,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            opt_path,
        )

        latest_path = os.path.join(self.config["out_dir"], "latest.ckpt")
        os.system(f"echo {it:06d} > {latest_path}")

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args["iterations"]))
        pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)

        while True:
            self.epoch += 1
            pbar.set_postfix(epoch=self.epoch)
            self.prefetcher.reset()
            self._train_epoch(pbar)
            # self.validate()
            self.scheduler.step()
            if self.iteration > self.train_args["iterations"]:
                break
        print("\nTraining complete.")

    def _train_epoch(self, pbar):
        """
        Process input and calculate loss every training epoch
        """

        device = self.device
        train_data = self.prefetcher.next()
        while train_data is not None:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                self.iteration += 1
                gt_frames, resized_gt_frames, blurry_frames, blur_maps, _ = (
                    train_data
                )
                gt_frames, resized_gt_frames, blurry_frames, blur_maps = (
                    gt_frames.to(device),
                    resized_gt_frames.to(device),
                    blurry_frames.to(device),
                    blur_maps.to(device).float(),
                )
                depths = []


                # Create Depth map of blurred images
                for batch in blurry_frames:
                    for blurred_img in batch:
                        depth = self.depth_model.infer_image(blurred_img.permute(1, 2, 0).cpu().numpy())
                        depth = 1 - ((depth - depth.min()) / (depth.max() - depth.min()))
                        depths.append(torch.tensor(depth).unsqueeze(0))
                depths = torch.stack(depths).to(device).unsqueeze(0)
                

                l_t = self.num_local_frames
                b, t, c, h, w = blurry_frames.size()


                gt_local_frames = resized_gt_frames[
                    :,
                    :l_t,
                ]
                binary_masks = (blur_maps > torch.min(blur_maps)).float()
                local_masks = binary_masks[
                    :,
                    :l_t,
                ].contiguous()

                masked_local_frames = blurry_frames[
                    :,
                    :l_t,
                ]

                # Get GT Optical Flow
                blurry_flows_bi = self.raft(masked_local_frames)

                # ---- Image Propagation ----
                prop_imgs, updated_local_masks = self.model.img_propagation(
                    masked_local_frames,
                    blurry_flows_bi,
                    local_masks,
                    interpolation=self.interp_mode,
                )
                updated_binary_masks = binary_masks.clone()
                updated_binary_masks[
                    :,
                    :l_t,
                ] = updated_local_masks.view(b, l_t, 1, h, w)
                updated_frames = blurry_frames.clone()
                prop_local_frames = (
                    gt_local_frames * (1 - local_masks)
                    + prop_imgs.view(b, l_t, 3, h, w) * local_masks
                )
                updated_frames[
                    :,
                    :l_t,
                ] = prop_local_frames

                # ---- Transformer + Super Resolution ----
                ori_pred_imgs = self.model(
                    updated_frames,
                    depths,
                    blurry_flows_bi,
                    blur_maps,
                    updated_binary_masks,
                    l_t,
                )
                pred_imgs = torch.stack(
                    [
                        transforms.Resize(size=(h, w), antialias=None)(batch)
                        for batch in ori_pred_imgs
                    ]
                )
                pred_imgs = pred_imgs.view(b, -1, c, h, w)

                # get the local frames
                pred_local_frames = pred_imgs[
                    :,
                    :l_t,
                ]

                # ---- Loss Calculation ----

                # Student l1 loss
                # hole_loss = self.l1_loss(
                #     pred_imgs * binary_masks, resized_frames * binary_masks
                # )
                # hole_loss = (
                #     hole_loss
                #     / torch.mean(binary_masks)
                #     * self.config["losses"]["hole_weight"]
                # )
                # valid_loss = self.l1_loss(
                #     pred_imgs * (1 - binary_masks), resized_frames * (1 - binary_masks)
                # )
                # valid_loss = (
                #     valid_loss
                #     / torch.mean(1 - binary_masks)
                #     * self.config["losses"]["valid_weight"]
                # )
                # self.add_summary(self.writer, "loss/hole_loss", hole_loss.item())
                # self.add_summary(self.writer, "loss/valid_loss", valid_loss.item())

                # # Super Resolution Loss
                # if self.config["losses"]["sr_weight"] > 0:
                #     sr_loss = (
                #         self.charbonnier_loss(ori_pred_imgs, frames)
                #         * self.config["losses"]["sr_weight"]
                #     )
                #     self.add_summary(self.writer, "loss/sr_loss", sr_loss.item())
                # else:
                #     sr_loss = 0

                # total_loss = sr_loss + hole_loss + valid_loss
                total_loss = self.l1_loss(ori_pred_imgs, gt_frames)
                self.add_summary(self.writer, "loss/total_loss", total_loss.item())                
                self.add_summary(self.writer, "loss/learning_rate", self.scheduler.get_last_lr()[0])

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            # with self.warmup_scheduler.dampening():
            self.scaler.update()
            self.optimizer.zero_grad()
            

            # write images to tensorboard
            if self.iteration % 250 == 0:
                # img to cpu
                gt_local_frames_cpu = (
                    (gt_local_frames.view(b, -1, 3, h, w) + 1) / 2.0
                ).cpu()
                masked_local_frames = (
                    (masked_local_frames.view(b, -1, 3, h, w) + 1) / 2.0
                ).cpu()
                prop_local_frames_cpu = (
                    (prop_local_frames.view(b, -1, 3, h, w) + 1) / 2.0
                ).cpu()
                pred_local_frames_cpu = (
                    (pred_local_frames.view(b, -1, 3, h, w) + 1) / 2.0
                ).cpu()
                img_results = torch.cat(
                    [
                        masked_local_frames[0][0],
                        prop_local_frames_cpu[0][0],
                        pred_local_frames_cpu[0][0],
                        gt_local_frames_cpu[0][0],
                    ],
                    1,
                )
                img_results = torchvision.utils.make_grid(
                    img_results, nrow=1, normalize=True
                )
                if self.writer is not None:
                    self.writer.add_image(
                        f"img/img:input-prop-out-gt (0)", img_results, self.iteration
                    )

                img_results = torch.cat(
                    [
                        masked_local_frames[0][-1],
                        prop_local_frames_cpu[0][-1],
                        pred_local_frames_cpu[0][-1],
                        gt_local_frames_cpu[0][-1],
                    ],
                    1,
                )
                img_results = torchvision.utils.make_grid(
                    img_results, nrow=1, normalize=True
                )
                if self.writer is not None:
                    self.writer.add_image(
                        f"img/img:input-prop-out-gt (-1)", img_results, self.iteration
                    )

                # flow to cpu
                flows_forward_cpu = flow_to_image(blurry_flows_bi[0][0]).cpu()
                flow_results = torch.cat(
                    [
                        flows_forward_cpu[0],
                    ],
                    1,
                )
                if self.writer is not None:
                    self.writer.add_image(
                        "img/flow:masked-pred-gt", flow_results, self.iteration
                    )

                # depth, blur, binary mask and updated binary mask to cpu
                binary_results = torch.cat(
                    [
                        blur_maps[0][0].cpu(),
                        binary_masks[0][0].cpu(),
                        updated_binary_masks[0][0].cpu(),
                        depths[0][0].cpu(),
                    ],
                    1,
                )
                if self.writer is not None:
                    self.writer.add_image(
                        "img/bin:blur-mask-up_mask-depth",
                        binary_results,
                        self.iteration,
                    )

            # console logs
            pbar.update(1)
            pbar.set_description((f"LR: {self.scheduler.get_last_lr()[0]} Loss: {total_loss.item():.3f}"))

            # saving models
            if self.iteration % self.train_args["save_freq"] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args["iterations"]:
                break

            train_data = self.prefetcher.next()


    # def validate(self):

    #     self.valid_loader = "T:/ProPainter Datasets/davis/blur_tests"

    #     self.model.eval()
    #     psnr_list = []

    #     pbar = tqdm(os.listdir(self.valid_loader))
    #     for video_name in pbar:
    #         blurry_frames, depths, blur_maps, fps = read_from_videos(self.valid_loader, video_name)

    #         # Preprocess frames
    #         blurry_frames = (blurry_frames / 255 * 2) - 1  # norm to -1, 1
    #         blurry_frames = blurry_frames[:, :3, :, :]  # only use RGB channels
    #         depths = depths[:, 0, :, :].unsqueeze(1) / 255.0  # norm to 0, 1
    #         blur_maps = blur_maps[:, 0, :, :].unsqueeze(1) / 255.0  # norm to 0, 1

    #         # Move to device (GPU)
    #         blurry_frames, depths, blur_maps = (
    #             blurry_frames.unsqueeze(0).to(self.device),
    #             depths.unsqueeze(0).to(self.device),
    #             blur_maps.unsqueeze(0).to(self.device),
    #         )

    #         binary_masks = (blur_maps > 0.1).float()

    #         with torch.no_grad():
    #             comp_frames = self.model(blurry_frames, depths, blur_maps, binary_masks)

    #         # Compute PSNR
    #         for comp_frame, gt_frame in zip(comp_frames, blurry_frames):
    #             psnr, _ = calc_psnr_and_ssim(comp_frame.cpu().numpy(), gt_frame.cpu().numpy())
    #             psnr_list.append(psnr)

    #     return psnr_list
