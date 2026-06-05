# -*- coding: utf-8 -*-
import os
import cv2
import math
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import time

import torch
import lpips as lpips_lib
from torch.nn import functional

from core.dataset import TestDataset
from torch.utils.data import DataLoader
from core.metrics import calc_psnr_and_ssim, calculate_tof
from model.modules.flow_comp_raft import RAFT_bi
from model.dabit import DaBiT
from model.modules.depth_anything_v2.dpt import DepthAnythingV2


# Number of worker processes used for the (CPU-bound) PSNR/SSIM/tOF metrics.
METRIC_WORKERS = min(12, max(1, (os.cpu_count() or 2) - 2))


def _compute_chunk_metrics(task):
    """Worker: compute PSNR/SSIM/tOF for a contiguous slice of frames.

    `comp_sub`/`gt_sub` are RGB float32 arrays in [0, 255]. When `has_prev` is
    True the slice is prefixed with one extra context frame (global `start - 1`)
    so tOF — which compares consecutive frames — can be evaluated across the
    chunk boundary. tOF is undefined for global frame 0 (first frame skipped,
    as in FMA-Net). Returns a list of (global_idx, psnr, ssim, tof_or_None).
    """
    cv2.setNumThreads(1)  # avoid oversubscription across worker processes
    comp_sub, gt_sub, start, has_prev = task
    offset = 1 if has_prev else 0

    out = []
    for p in range(offset, len(comp_sub)):
        g = start + (p - offset)
        psnr, ssim = calc_psnr_and_ssim(comp_sub[p], gt_sub[p])
        tof = None
        if p >= 1:  # a previous frame is available in this slice
            tof = calculate_tof(comp_sub[p - 1], comp_sub[p], gt_sub[p - 1], gt_sub[p])
        out.append((g, psnr, ssim, tof))
    return out


# def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
def get_ref_index(length, sample_length, local_idx):

        smaller_idx = sorted([i for i in range(length) if i < min(local_idx)])
        larger_idx = sorted([i for i in range(length) if i > max(local_idx)])

        total_available = len(smaller_idx) + len(larger_idx)
        smaller_samples = round(sample_length * len(smaller_idx) / total_available)
        larger_samples = sample_length - smaller_samples

        smaller_step = len(smaller_idx) // smaller_samples if smaller_samples > 0 else 0
        larger_step = len(larger_idx) // larger_samples if larger_samples > 0 else 0

        ref_idx = []
        if smaller_samples > 0:
            smaller_step = len(smaller_idx) // smaller_samples
            start_idx = smaller_step // 2 
            ref_idx.extend([smaller_idx[start_idx + i * smaller_step] for i in range(smaller_samples)])

        if larger_samples > 0:
            larger_step = len(larger_idx) // larger_samples
            start_idx = (larger_step - 1) // 2 
            ref_idx.extend([larger_idx[start_idx + i * larger_step] for i in range(larger_samples)])

        return sorted(ref_idx)






if __name__ == "__main__":

    # Input sizes are constant across DAVIS-Blur, so let cuDNN pick the fastest
    # convolution algorithms once and reuse them.
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using ", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU instead")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./blur_tests",
        help="Path of the input video or image folder.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./results",
        help="Output folder. Default: results",
    )
    parser.add_argument(
        "--ref_stride", type=int, default=10, help="Stride of global reference frames."
    )
    parser.add_argument(
        "--neighbor_length",
        type=int,
        default=10,
        help="Length of local neighboring frames.",
    )
    parser.add_argument(
        "--subvideo_length",
        type=int,
        default=80,
        help="Length of sub-video for long video inference.",
    )
    parser.add_argument(
        "--raft_iter", type=int, default=20, help="Iterations for RAFT inference."
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save refocused frames to the output folder (off by default; "
        "skipping the per-frame PNG writes makes evaluation much faster).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./weights/dabit.pth",
        help="Path to the trained DaBiT model.",
    )

    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #############################################
    # Set up RAFT
    ##############################################

    raft = RAFT_bi(model_path="./weights/raft.pth", device=device).to(device)

    #############################################
    # Set up DepthAnythingV2
    ##############################################

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
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
    depth_model = depth_model.to(device).eval()

    ##############################################
    # Set up DaBiT
    ##############################################

    dabit = DaBiT()
    # dabit.load_state_dict(torch.load("./weights/dabit.pth", weights_only=True))
    dabit.load_state_dict(torch.load(args.model, weights_only=True))
    for p in dabit.parameters():
        p.requires_grad = False
    dabit.to(device)
    dabit.eval()

    ##############################################
    # Set up LPIPS metric (AlexNet backbone)
    ##############################################

    lpips_model = lpips_lib.LPIPS(net="alex").to(device).eval()

    # Pure inference: disable autograd everywhere so none of RAFT / Depth Anything
    # / DaBiT / LPIPS build throwaway graphs (faster, lower VRAM, identical values).
    torch.set_grad_enabled(False)

    ##############################################
    # Begin Testing Loop
    ##############################################

    metrics = ["PSNR", "SSIM", "LPIPS", "tOF", "Time"]
    results_dict = {k: [] for k in metrics}
    results_dict["Frames"] = []
    logfile = open("./results/results.txt", "a")
    logfile.write(f"Model: {args.model}")

    # Background-process pool for the CPU-bound metrics (PSNR/SSIM/tOF). The
    # workers are pure NumPy/OpenCV and never touch CUDA, so 'fork' is safe and
    # lets them inherit the already-imported metric functions (no torch re-import).
    metric_pool = ProcessPoolExecutor(
        max_workers=METRIC_WORKERS, mp_context=mp.get_context("fork")
    )

    # The DataLoader decodes/resizes the next video in background worker
    # processes (TestDataset now returns CPU tensors) while the GPU is busy.
    test_ds = TestDataset(device, args.input)
    valid_loader = DataLoader(
        dataset=test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    pbar = tqdm(sorted(os.listdir(args.input)))
    for gt_tensors, input_tensors, blur_maps, video_name in valid_loader:
        video_start_time = time.time()

        # Move the inputs the GPU pipeline needs onto the device; ground-truth
        # frames stay on the CPU (only used for metrics).
        input_tensors = input_tensors.to(device, non_blocking=True)
        blur_maps = blur_maps.to(device, non_blocking=True)

        # Preprocess frames
        input_size = input_tensors[0].shape[-2:]  # height, width
        h, w = input_size
        out_h, out_w = input_size[0] * 2, input_size[1] * 2


        # Create binary masks and dilated masks
        binary_masks = (
            (blur_maps[0] > torch.min(blur_maps)).float()
        ).unsqueeze(0)
        binary_masks = torch.stack(
            [functional.max_pool2d(batch, kernel_size=3, stride=1, padding=1) for batch in binary_masks]
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
                        flows_f, flows_b = raft(
                            input_tensors[:, f:end_f], iters=args.raft_iter
                        )
                else:
                    with torch.no_grad():
                        flows_f, flows_b = raft(
                            input_tensors[:, f - 1 : end_f], iters=args.raft_iter
                        )

                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()

            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            flows_bi = (gt_flows_f, gt_flows_b)
        else:
            with torch.no_grad():
                flows_bi = raft(input_tensors, iters=args.raft_iter)
            torch.cuda.empty_cache()

        # ---- image propagation ----
        subvideo_length_img_prop = min(100, args.subvideo_length)

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
                    dabit.img_propagation(
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
            prop_imgs, updated_local_masks = dabit.img_propagation(
                input_tensors, flows_bi, binary_masks, interpolation="nearest"
            )
            updated_frames = (
                input_tensors * (1 - binary_masks)
                + prop_imgs.view(b, t, 3, h, w) * binary_masks
            )
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()

        ##############################################
        # Depth prediction (precomputed once per frame)
        ##############################################
        # Depth Anything is applied per frame with a per-frame min-max
        # normalisation, so a frame's depth is independent of which inference
        # window it lands in. The original code recomputed it inside the sliding
        # window (every frame several times over); caching it here is bit-for-bit
        # identical but evaluates each frame's depth exactly once.
        depth_bs = 12
        frame_depths = [None] * video_length
        with torch.no_grad():
            for s in range(0, video_length, depth_bs):
                e = min(video_length, s + depth_bs)
                depth_batch = depth_model.preprocess_tensor(updated_frames[0, s:e])
                depth_batch = depth_model.forward(depth_batch).unsqueeze(1)
                depth_batch = functional.interpolate(
                    depth_batch, (h, w), mode="bilinear", align_corners=True
                )
                for k in range(depth_batch.shape[0]):
                    depth = depth_batch[k]
                    depth = 1 - ((depth - depth.min()) / (depth.max() - depth.min()))
                    frame_depths[s + k] = depth
        frame_depths = torch.stack(frame_depths)  # [video_length, 1, h, w]
        torch.cuda.empty_cache()

        comp_frames = [None] * video_length

        neighbor_stride = args.neighbor_length // 2
        if video_length > args.subvideo_length:
            ref_num = args.subvideo_length // args.ref_stride
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
            selected_pred_flows_bi = raft(selected_imgs, iters=args.raft_iter)

            # ---- depth prediction (cached, see precompute above) ----
            selected_depths = frame_depths[neighbor_ids + ref_ids].unsqueeze(0)


            l_t = len(neighbor_ids)

            # print_summary(selected_imgs, "selected_imgs")
            # print_summary(selected_depths, "selected_depths")
            # print_summary(selected_blur_maps, "selected_blur_maps")
            # print_summary(selected_update_masks, "selected_update_masks")
            # print("local_index: ", local_index)

            with torch.no_grad():
                ori_pred_img = dabit(
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



        # save videos
        if args.save_videos:

            # Create output directory
            save_root = f"./results/{str(video_name[0])}"
            os.makedirs(save_root, exist_ok=True)

            for comp_frame in range(len(comp_frames)):

                # Save images
                img_for_save = cv2.cvtColor(comp_frames[comp_frame], cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(save_root, f"frame_{comp_frame}.png"), img_for_save)


            # masked_frame_for_save = []
            # for i in range(len(input_tensors[0])):
            #     img = np.array(
            #         (input_tensors[0][i]).cpu().permute(1, 2, 0) * 255
            #     )
            #     masked_frame_for_save.append(img.astype(np.uint8))

            # imageio.mimwrite(
            #     os.path.join(save_root, "masked_in.mp4"),
            #     masked_frame_for_save,
            # )

            # imageio.mimwrite(
            #     os.path.join(save_root, "refocused_out.mp4"),
            #     comp_frames,
            # )

            # pbar.set_description(f"Videos saved.")
            # torch.cuda.empty_cache()

        ###########################################
        # compute metrics
        ###########################################
        pbar.set_description(f"Computing metrics for {video_name[0]}")
        gt_frames = gt_tensors[0].permute(0, 2, 3, 1).cpu().numpy()

        # Build the comparison arrays once (RGB float32, [0, 255]); identical
        # preprocessing to before — only the per-frame loop is gone.
        comp_arrs, gt_arrs = [], []
        for comp_frame, gt_frame in zip(comp_frames, gt_frames):
            comp_arrs.append(
                cv2.resize(
                    cv2.cvtColor(comp_frame, cv2.COLOR_BGR2RGB),
                    (gt_frame.shape[1], gt_frame.shape[0]),
                ).astype(np.float32)
            )
            gt_arrs.append((gt_frame * 255).astype(np.float32))

        # ---- LPIPS: batched on the GPU (per-frame value unchanged) ----
        lpips_list = []
        with torch.no_grad():
            for s in range(0, len(comp_arrs), 16):
                e = min(len(comp_arrs), s + 16)
                c = torch.from_numpy(np.stack(comp_arrs[s:e]) / 255.0).permute(0, 3, 1, 2).float().to(device)
                g = torch.from_numpy(np.stack(gt_arrs[s:e]) / 255.0).permute(0, 3, 1, 2).float().to(device)
                lpips_list.extend(lpips_model(c, g, normalize=True).view(-1).cpu().tolist())

        # ---- PSNR / SSIM / tOF: parallelised across CPU cores ----
        n_workers = max(1, min(METRIC_WORKERS, video_length))
        chunk = math.ceil(video_length / n_workers)
        tasks = []
        for s in range(0, video_length, chunk):
            e = min(video_length, s + chunk)
            has_prev = s > 0
            lo = s - 1 if has_prev else s  # one frame of context for tOF
            tasks.append((comp_arrs[lo:e], gt_arrs[lo:e], s, has_prev))

        psnr_list = [0.0] * video_length
        ssim_list = [0.0] * video_length
        tof_list = []
        for chunk_out in metric_pool.map(_compute_chunk_metrics, tasks):
            for g, psnr, ssim, tof in chunk_out:
                psnr_list[g] = psnr
                ssim_list[g] = ssim
                if tof is not None:
                    tof_list.append(tof)

        results_dict["PSNR"].append(np.mean(psnr_list))
        results_dict["SSIM"].append(np.mean(ssim_list))
        results_dict["LPIPS"].append(np.mean(lpips_list))
        results_dict["tOF"].append(np.mean(tof_list))

        summary = (
            f"PSNR: {np.mean(psnr_list):.2f}, SSIM: {np.mean(ssim_list):.4f}, "
            f"LPIPS: {np.mean(lpips_list):.4f}, tOF: {np.mean(tof_list):.4f}, "
            f"Time: {avg_runtime:.4f}"
        )
        pbar.update(1)
        pbar.write(f"{video_name[0]} {summary}")
        logfile.write("\n" + "{:<15s} -- {}".format(f"{video_name[0]}", summary))


    msg = (
        "\n"
        + "{:<15s} -- {}".format(
            "Average", {k: round(np.mean(results_dict[k]), 4) for k in metrics}
        )
        + "\n \n"
    )

    print(msg, end="")
    logfile.write(msg)
    logfile.close()
    metric_pool.shutdown()


