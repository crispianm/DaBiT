import os
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.loss import AdversarialLoss, PerceptualLoss, LPIPSLoss, CharbonnierLoss
from core.dataset import *

from model.modules.flow_comp_raft import RAFT_bi, FlowLoss, EdgeLoss
from model.recurrent_flow_completion import RecurrentFlowCompleteNet


from RAFT.utils.flow_viz_pt import flow_to_image


class Trainer:
    def __init__(self, config, prefetcher, model, start_epoch=0):

        self.l1_loss = nn.L1Loss()
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
        self.fix_raft = RAFT_bi(device=self.device)
        self.fix_flow_complete = RecurrentFlowCompleteNet(
            "C:/Users/wg19671/repos/DepthPainter/experiments_model/recurrent_flow_completion_train_flowcomp/gen_108000.pth"
        )
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(self.device)
        self.fix_flow_complete.eval()

        self.interp_mode = self.config["interp_mode"]
        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
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

        self.optimizer = torch.optim.Adam(
            optim_params,
            betas=(self.config["trainer"]["beta1"], self.config["trainer"]["beta2"]),
        )

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config["trainer"]["scheduler"]
        scheduler_type = scheduler_opt.pop("type")

        if scheduler_type in ["MultiStepLR", "MultiStepRestartLR"]:
            self.scheG = MultiStepRestartLR(
                self.optimizer,
                milestones=scheduler_opt["milestones"],
                gamma=scheduler_opt["gamma"],
            )

        elif scheduler_type == "CosineAnnealingRestartLR":
            self.scheG = CosineAnnealingRestartLR(
                self.optimizer,
                periods=scheduler_opt["periods"],
                restart_weights=scheduler_opt["restart_weights"],
                eta_min=scheduler_opt["eta_min"],
            )

        else:
            raise NotImplementedError(
                f"Scheduler {scheduler_type} is not implemented yet."
            )

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
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
            model_data = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model_data)

            data_opt = torch.load(opt_path, map_location=self.device)
            self.optimizer.load_state_dict(data_opt["optimG"])

            self.epoch = data_opt["epoch"]
            self.iteration = data_opt["iteration"]
        else:
            model_path = self.config["trainer"].get("model_path", None)
            opt_path = self.config["trainer"].get("opt_path", None)
            if model_path is not None:
                print(f"Loading Gen-Net from {model_path}")
                model_data = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(model_data)

                if opt_path is not None:
                    data_opt = torch.load(opt_path, map_location=self.device)
                    self.optimizer.load_state_dict(data_opt["optimG"])
                    self.scheG.load_state_dict(data_opt["scheG"])

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
                "optimG": self.optimizer.state_dict(),
                "scheG": self.scheG.state_dict(),
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
            self.prefetcher.reset()
            self._train_epoch(pbar)
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

            self.iteration += 1
            frames, blurry_frames, depths, blur_maps, flows_f, flows_b, _ = train_data
            frames, blurry_frames, depths, blur_maps = (
                frames.to(device),
                blurry_frames.to(device),
                depths.to(device).float(),
                blur_maps.to(device).float(),
            )
            
            # frames: [b, t, c, ori_h, ori_w]
            # blurry_frames: [b, t, c, h, w]
            # masks: [b, t, 1, h, w]
            # depths: [b, t, 1, h, w]

            l_t = self.num_local_frames
            b, t, c, ori_h, ori_w = frames.size()
            b, t, c, h, w = blurry_frames.size()

            resized_frames = torch.stack(
                [
                    transforms.Resize(size=(h, w), antialias=None)(batch)
                    for batch in frames
                ]
            )
            gt_local_frames = resized_frames[
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
            if flows_f[0] == "None" or flows_b[0] == "None":
                gt_flows_bi = self.fix_raft(gt_local_frames)
            else:
                gt_flows_bi = (flows_f.to(device), flows_b.to(device))

            # ---- Complete Flow ----
            pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(
                blurry_frames[:, :l_t], gt_flows_bi, local_masks
            )
            pred_flows_bi = self.fix_flow_complete.combine_flow(
                gt_flows_bi, pred_flows_bi, local_masks
            )

            # ---- Image Propagation ----
            prop_imgs, updated_local_masks = self.model.img_propagation(
                masked_local_frames,
                pred_flows_bi,
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

            # ---- Feature Propagation + Transformer + Super Resolution ----
            ori_pred_imgs = self.model(
                updated_frames,
                depths,
                pred_flows_bi,
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
            self.optimizer.zero_grad()

            # Student l1 loss
            hole_loss = self.l1_loss(
                pred_imgs * binary_masks, resized_frames * binary_masks
            )
            hole_loss = (
                hole_loss
                / torch.mean(binary_masks)
                * self.config["losses"]["hole_weight"]
            )
            valid_loss = self.l1_loss(
                pred_imgs * (1 - binary_masks), resized_frames * (1 - binary_masks)
            )
            valid_loss = (
                valid_loss
                / torch.mean(1 - binary_masks)
                * self.config["losses"]["valid_weight"]
            )
            self.add_summary(self.writer, "loss/hole_loss", hole_loss.item())
            self.add_summary(self.writer, "loss/valid_loss", valid_loss.item())

            # Super Resolution Loss
            if self.config["losses"]["sr_weight"] > 0:
                sr_loss = (
                    self.charbonnier_loss(ori_pred_imgs, frames)
                    * self.config["losses"]["sr_weight"]
                )
                self.add_summary(self.writer, "loss/sr_loss", sr_loss.item())
            else:
                sr_loss = 0

            total_loss = hole_loss + valid_loss + sr_loss
            self.add_summary(self.writer, "loss/z_total_loss", total_loss.item())
            total_loss.backward()
            self.optimizer.step()

            self.update_learning_rate()

            # write images to tensorboard
            if self.iteration % 100 == 0:
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
                gt_flows_forward_cpu = flow_to_image(gt_flows_bi[0][0]).cpu()
                masked_flows_forward_cpu = (
                    gt_flows_forward_cpu[0] * (1 - local_masks[0][0].cpu())
                    + local_masks[0][0].cpu()
                ).to(gt_flows_forward_cpu)
                pred_flows_forward_cpu = flow_to_image(pred_flows_bi[0][0]).cpu()

                flow_results = torch.cat(
                    [
                        masked_flows_forward_cpu,
                        pred_flows_forward_cpu[0],
                        gt_flows_forward_cpu[0],
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
            pbar.set_description((f"LR: {self.get_lr()} Loss: {total_loss.item():.3f}"))

            # saving models
            if self.iteration % self.train_args["save_freq"] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args["iterations"]:
                break

            train_data = self.prefetcher.next()
