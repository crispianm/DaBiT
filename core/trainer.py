import os
import gc
import glob
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from core.loss import AdversarialLoss, PerceptualLoss, LPIPSLoss, CharbonnierLoss
from core.dataset import *

from model.modules.flow_comp_raft import RAFT_bi
from model.modules.depth_anything_v2.dpt import DepthAnythingV2

from test_dabit import get_ref_index

from core.metrics import calc_psnr_and_ssim
from RAFT.utils.flow_viz_pt import flow_to_image

def print_summary(name, namestring):
    print(
        f"{namestring} shape: ",
        name.shape,
        "\n\t(Min, Max) = (",
        name.min().item(),
        ", ",
        name.max().item(),
        ")",
    )




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
        self.scaler = torch.GradScaler("cuda", enabled=True)
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
            self.optimizer, step_size=100000, gamma=0.1
        )

    def add_summary(self, writer, name, val):
        """
        Add tensorboard summary.

        """
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        writer.add_scalar(name, self.summary[name], self.iteration)
        self.summary[name] = 0

    def load(self):
        """Load DaBiT model from checkpoint."""
        # get the latest checkpoint
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
            model_data = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
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
                print(f"Loading model from {model_path}")
                model_data = torch.load(
                    model_path, map_location=self.device, weights_only=True
                )
                self.model.load_state_dict(model_data)

                if opt_path is not None:
                    data_opt = torch.load(
                        opt_path, map_location=self.device, weights_only=True
                    )
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
            torch.cuda.empty_cache()
            gc.collect()
            # self._train_epoch(pbar)
            self.validate()
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
                            depth = 1 - ((depth - depth.min()) / (depth.max() - depth.min()))
                            depths.append(torch.tensor(depth).unsqueeze(0))
                    depths = torch.stack(depths).to(device).unsqueeze(0)

                # Create Binary Masks
                binary_masks = (blur_maps > torch.min(blur_maps)).float()

                # Get GT Optical Flow
                blurry_flows_bi = self.raft(blurry_frames)


                # ---- Image Propagation ----
                prop_frames, prop_masks = self.model.img_propagation(
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
                    [
                        transforms.Resize(size=(h, w), antialias=None)(batch)
                        for batch in ori_pred_frames
                    ]
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
                self.add_summary(self.writer, "loss/blur_loss", blur_loss.item())
                self.add_summary(self.writer, "loss/clear_loss", clear_loss.item())

                # Super Resolution Loss
                if self.config["losses"]["sr_weight"] > 0:
                    sr_loss = (
                        self.charbonnier_loss(ori_pred_frames, gt_frames)
                        * self.config["losses"]["sr_weight"]
                    )
                    self.add_summary(self.writer, "loss/sr_loss", sr_loss.item())
                else:
                    sr_loss = 0

                total_loss = sr_loss + blur_loss + clear_loss
                # total_loss = self.l1_loss(ori_pred_frames, gt_frames)

                if self.iteration % self.train_args["log_freq"] == 0:
                    self.add_summary(self.writer, "loss/total_loss", total_loss.item())
                    self.add_summary(
                        self.writer, "loss/learning_rate", self.scheduler.get_last_lr()[0]
                    )
                
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            ####################################################
            # Write images to tensorboard
            ####################################################
            if self.iteration % self.train_args["img_freq"] == 0:
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
                torch.cuda.empty_cache()
                pbar.set_description(
                    (f"LR: {self.scheduler.get_last_lr()[0]} Loss: {total_loss.item():.3f}")
                )

            # saving models
            if self.iteration % self.train_args["save_freq"] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args["iterations"]:
                break

            train_data = self.prefetcher.next()


    def validate(self):

        metrics = ["PSNR", "SSIM", "LPIPS", "Time"]
        results_dict = {k: [] for k in metrics}
        results_dict["Frames"] = []
        logfile = open("./results/results.txt", "a")
        logfile.write(f"Iteration: {self.iteration}")

        print("Validating model:")

        test_ds = TestDataset(self.device, "./blur_tests")
        self.valid_loader = DataLoader(
            dataset=test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        self.model.eval()

        pbar = tqdm(sorted(os.listdir("./blur_tests")))


        for gt_tensors, input_tensors, blur_maps, video_name in self.valid_loader:
            video_start_time = time.time()

            # Preprocess frames
            input_size = input_tensors[0].shape[-2:]  # height, width
            h, w = input_size
            out_h, out_w = input_size[0] * 2, input_size[1] * 2


            # Create binary masks and dilated masks
            binary_masks = (
                (blur_maps[0] > torch.min(blur_maps)).float()
            ).unsqueeze(0)
            binary_masks = torch.stack(
                [nn.functional.max_pool2d(batch, kernel_size=3, stride=1, padding=1) for batch in binary_masks]
            )
            binary_masks = torch.zeros_like(binary_masks) + 1

            video_length = input_tensors.shape[1]
            pbar.set_description(f"Processing: {video_name[0]} ({video_length} frames)")

            ##############################################
            # Image Propagation
            ##############################################
            
            short_clip_len = 12

            # use fp32 for RAFT
            if video_length > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        with torch.no_grad():
                            flows_f, flows_b = self.raft(
                                input_tensors[:, f:end_f], iters=20
                            )
                    else:
                        with torch.no_grad():
                            flows_f, flows_b = self.raft(
                                input_tensors[:, f - 1 : end_f], iters=20
                            )

                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()

                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                flows_bi = (gt_flows_f, gt_flows_b)
            else:
                with torch.no_grad():
                    flows_bi = self.raft(input_tensors, iters=20)
                torch.cuda.empty_cache()

            # ---- image propagation ----
            subvideo_length_img_prop = min(100, 80)

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
                        self.model.img_propagation(
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
                prop_imgs, updated_local_masks = self.model.img_propagation(
                    input_tensors, flows_bi, binary_masks, interpolation="nearest"
                )
                updated_frames = (
                    input_tensors * (1 - binary_masks)
                    + prop_imgs.view(b, t, 3, h, w) * binary_masks
                )
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()

            comp_frames = [None] * video_length

            neighbor_stride = 5
            if video_length > 80:
                ref_num = 6
            else:
                ref_num = -1

            ##############################################
            # Run Model
            ##############################################
            neighbor_ids = None
            for f in tqdm(range(0, video_length, neighbor_stride)[:-1], leave=False):


                if f + 10 > video_length:
                    start = video_length - 10
                    end = video_length
                else:
                    start = f
                    end = f + 10
                
                current_list = list(range(start, end))
                if current_list != neighbor_ids: 
                    neighbor_ids = current_list


                ref_ids = get_ref_index(video_length, 6, neighbor_ids)
                selected_index = sorted(neighbor_ids + ref_ids)
                local_index = [i for i in range(len(selected_index)) if selected_index[i] in neighbor_ids]


                selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
                selected_blur_maps = blur_maps[:, neighbor_ids + ref_ids, :, :, :]
                selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
                selected_pred_flows_bi = self.raft(selected_imgs, iters=20)


                # ---- depth prediction ----
                selected_depths = []
                with torch.no_grad():
                    for batch in selected_imgs:
                        depth_batch = self.depth_model.preprocess_tensor(batch)
                        depth_batch = self.depth_model.forward(depth_batch).unsqueeze(1)
                        depth_batch = nn.functional.interpolate(depth_batch, (h, w), mode="bilinear", align_corners=True)
                        for depth in depth_batch:
                            depth = 1 - ((depth - depth.min()) / (depth.max() - depth.min()))
                            selected_depths.append(depth)
                    selected_depths = torch.stack(selected_depths).to(self.device).unsqueeze(0)


                l_t = len(neighbor_ids)

                # print_summary(selected_imgs, "selected_imgs")
                # print_summary(selected_depths, "selected_depths")
                # print_summary(selected_blur_maps, "selected_blur_maps")
                # print_summary(selected_update_masks, "selected_update_masks")
                # print("local_index: ", local_index)

                with torch.no_grad():
                    ori_pred_img = self.model(
                        selected_imgs,
                        selected_depths,
                        selected_pred_flows_bi,
                        selected_blur_maps,
                        selected_update_masks,
                        l_t,
                    )
                pred_img = ori_pred_img.view(-1, 3, out_h, out_w).permute(0, 2, 3, 1)
                pred_img = (pred_img).clamp(0, 1)


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
            video_end_time = time.time()

            avg_runtime = (video_end_time - video_start_time) / video_length
            results_dict["Time"].append(avg_runtime)
            pbar.write(f"{video_name[0]} Average Runtime (s): {avg_runtime:.4f}")

            ###########################################
            # compute metrics
            ###########################################
            pbar.set_description(f"Computing metrics for {video_name[0]}")
            gt_frames = gt_tensors[0].permute(0,2,3,1).cpu().numpy()

            psnr_list, ssim_list, lpips_list = [], [], []
            for comp_frame, gt_frame in tqdm(zip(comp_frames, gt_frames), leave=False):

                comp_frame = cv2.resize(cv2.cvtColor(comp_frame, cv2.COLOR_BGR2RGB), (gt_frame.shape[1], gt_frame.shape[0])).astype(np.float32)
                gt_frame = (gt_frame * 255).astype(np.float32)
                cv2.imwrite("comp_frame.png", comp_frame)
                cv2.imwrite("gt_frame.png", gt_frame)

                psnr, ssim = calc_psnr_and_ssim(comp_frame, gt_frame)
                lpips = 0
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpips)


    
            results_dict["PSNR"].append(np.mean(psnr_list))
            results_dict["SSIM"].append(np.mean(ssim_list))
            results_dict["LPIPS"].append(np.mean(lpips_list))
            results_dict["Time"].append(avg_runtime)

            pbar.update(1)
            pbar.write(
                f"{video_name[0]} PSNR: {np.mean(psnr_list):.2f}, SSIM: {np.mean(ssim_list):.4f}, LPIPS: {np.mean(lpips_list):.4f}, Time: {avg_runtime:.4f}"
            )

            msg = (
                "\n" + 
                "{:<15s} -- {}".format(
                    f"{video_name[0]}",
                    f"PSNR: {np.mean(psnr_list):.2f}, SSIM: {np.mean(ssim_list):.4f}, LPIPS: {np.mean(lpips_list):.4f}, Time: {avg_runtime:.4f}",
                )
            )
            logfile.write(msg)

        msg = (
            "\n"
            + "{:<15s} -- {}".format(
                "Average", {k: round(np.mean(results_dict[k]), 4) for k in metrics}
            )
            + "\n \n"
        )

        self.writer.add_scalar("validation/avg_psnr", np.mean(results_dict["PSNR"]), self.iteration)
        self.writer.add_scalar("validation/avg_ssim", np.mean(results_dict["SSIM"]), self.iteration)

        print(msg, end="")
        logfile.write(msg)
        logfile.close()

