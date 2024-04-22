# -*- coding: utf-8 -*-
import os
import argparse
import imageio
import numpy as np
from tqdm import tqdm
import time

import torch
import torchvision
from torch.nn import functional
import matplotlib.pyplot as plt

from core.metrics import calc_psnr_and_ssim
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.depthpainter import DepthPainter

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


#  read frames from video
def read_from_videos(root_dir, video_name):
    """
    Read blurry_frames, depths, and masks from the specified root directory.

    Args:
        root_dir (str): The root directory containing the frames, depths, and masks folders.

    Returns:
        tuple: A tuple containing the following elements:
            - blurry_frames (torch.Tensor): A tensor containing the blurry_frames read from the frames folder.
            - depths (torch.Tensor): A tensor containing the depths read from the depths folder.
            - masks (torch.Tensor): A tensor containing the masks read from the masks folder.
            - fps (None): The frames per second (fps) of the video (currently set to None).
            - video_name (str): The name of the video (extracted from the root directory).

    """

    root_dir = os.path.join(root_dir, video_name)

    frame_path = os.path.join(root_dir, "frames")
    blurry_frames = []
    fr_lst = sorted(os.listdir(frame_path))
    for fr in fr_lst:
        blurry_frame = torchvision.io.read_image(os.path.join(frame_path, fr))
        blurry_frames.append(blurry_frame)

    depth_path = os.path.join(root_dir, "depths")
    depths = []
    depth_lst = sorted(os.listdir(depth_path))
    for fr in depth_lst:
        depth = torchvision.io.read_image(os.path.join(depth_path, fr))
        depths.append(depth)

    mask_path = os.path.join(root_dir, "masks")
    masks = []
    mask_lst = sorted(os.listdir(mask_path))
    for fr in mask_lst:
        mask = torchvision.io.read_image(os.path.join(mask_path, fr))
        masks.append(mask)

    fps = None

    return (
        torch.stack(blurry_frames),
        torch.stack(depths),
        torch.stack(masks),
        fps,
    )


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    """
    Get reference indices based on the given parameters.

    Parameters:
    - mid_neighbor_id (int): The ID of the mid neighbor.
    - neighbor_ids (list): List of neighbor IDs.
    - length (int): The total length.
    - ref_stride (int, optional): The stride between reference indices. Default is 10.
    - ref_num (int, optional): The number of reference indices to return. Default is -1, which means all available indices.

    Returns:
    - ref_index (list): List of reference indices.

    """
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


def get_frames(n, video_length, stride):
    """
    Returns a list of lists, where each inner list contains 'n' frame indices
    for a model to process, taking into account a desired overlap stride.
    Handles video lengths that are not perfect multiples of 'n'.

    Args:
        n (int): The number of frames to return in each group.
        video_length (int): The total number of frames in the video.
        stride (int): The amount of overlap between returned groups of frames.

    Returns:
        list: A list of lists, where each inner list contains 'n' frame indices.
    """

    frame_groups = []
    start_frame = 0

    while start_frame + n <= video_length:
        frame_groups.append(list(range(start_frame, start_frame + n)))
        start_frame += stride

    # Handle the remaining frames if video_length is not a multiple of n
    if start_frame < video_length:
        last_group_start = max(
            video_length - n, 0
        )  # Start the last group as late as possible
        frame_groups.append(list(range(last_group_start, video_length)))

    return frame_groups


def get_gt_frames(gt_dir, video_name):
    frame_path = os.path.join(gt_dir, video_name)
    frames = []
    fr_lst = sorted(os.listdir(frame_path))
    for fr in fr_lst:
        frame = plt.imread(os.path.join(frame_path, fr))
        if np.max(frame) > 1:
            frames.append((frame).astype(np.uint8))
        else:
            frames.append((frame * 255).astype(np.uint8))
    return frames


def dilate_mask(
    image, strel=torch.ones((9, 9)).to("cuda"), origin=(4, 4), border_value=0
):
    """
    Performs dilation operation on a given image using a structural element.

    Args:
        image (torch.Tensor): The input image tensor.
        strel (torch.Tensor): The structural element tensor.
        origin (tuple, optional): The origin coordinates of the structural element. Defaults to (0, 0).
        border_value (int, optional): The value to be used for padding the image. Defaults to 0.

    Returns:
        torch.Tensor: The dilated image tensor.

    Raises:
        ValueError: If the dimensions of the image and structural element do not match.

    Example:
        image = torch.tensor([[1, 0, 1],
                              [0, 1, 0],
                              [1, 0, 1]])
        strel = torch.tensor([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])
        dilated_image = dilation_pytorch(image, strel, origin=(1, 1))
        print(dilated_image)
        # Output: tensor([[1, 1, 1],
        #                 [1, 1, 1],
        #                 [1, 1, 1]])
    """

    image_pad = functional.pad(
        image,
        [
            origin[0],
            strel.shape[0] - origin[0] - 1,
            origin[1],
            strel.shape[1] - origin[1] - 1,
        ],
        mode="constant",
        value=border_value,
    )
    image_unfold = functional.unfold(image_pad, kernel_size=strel.shape)
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    sums = image_unfold + strel_flatten
    result, _ = sums.max(dim=1)

    return torch.reshape(result, image.shape)


if __name__ == "__main__":

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
        default="T:/ProPainter Datasets/davis",
        help="Path of the input video or image folder.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="T:/ProPainter Datasets/davis/results",
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
        default=True,
        help="Save inpainted videos to the output folder.",
    )

    args = parser.parse_args()

    #############################################
    # Set up RAFT and flow competition depthpainter
    ##############################################

    raft = RAFT_bi(model_path="./weights/raft-things.pth", device=device).to(device)

    flow_completion = RecurrentFlowCompleteNet()
    # flow_completion.load_state_dict(
    #     torch.load("./weights/recurrent_flow_completion.pth")
    # )
    flow_completion.load_state_dict(
        torch.load(
            "C:/Users/wg19671/repos/DepthPainter/experiments_model/recurrent_flow_completion_train_flowcomp/gen_025000.pth"
        )
    )
    for p in flow_completion.parameters():
        p.requires_grad = False
    flow_completion.to(device)
    flow_completion.eval()

    ##############################################
    # Set up DepthPainter depthpainter
    ##############################################

    depthpainter = DepthPainter()
    depthpainter.load_state_dict(torch.load("./weights/DepthPainter.pth"))
    # depthpainter.load_state_dict(
    #     torch.load(
    #         "C:/Users/wg19671/repos/DepthPainter/output/depthpainter_train_depthpainter/model_169000.pth"
    #     )
    # )
    for p in depthpainter.parameters():
        p.requires_grad = False
    depthpainter.to(device)
    depthpainter.eval()

    ##############################################
    # Begin Testing Loop
    ##############################################

    test_dir = os.path.join(args.input, "blur_tests_mini")

    metrics = ["PSNR", "SSIM"]
    results_dict = {k: [] for k in metrics}
    results_dict["Times"] = []
    results_dict["Frames"] = []
    logfile = open(os.path.join(args.output, "results.txt"), "a")

    pbar = tqdm(os.listdir(test_dir))

    for video_name in pbar:

        blurry_frames, depths, blur_maps, fps = read_from_videos(test_dir, video_name)

        # Preprocess frames
        blurry_frames = (blurry_frames / 255 * 2) - 1  # norm to -1, 1
        blurry_frames = blurry_frames[:, :3, :, :]  # only use RGB channels
        depths = depths[:, 0, :, :].unsqueeze(1) / 255.0  # norm to 0, 1
        blur_maps = blur_maps[:, 0, :, :].unsqueeze(1) / 255.0  # norm to 0, 1

        # resize = torchvision.transforms.Resize(size=(240, 432), antialias=None)
        # blurry_frames = resize(blurry_frames)
        # depths = resize(depths)
        # blur_maps = resize(blur_maps)

        input_size = blurry_frames[0].shape[-2:]  # height, width
        h, w = input_size

        out_h, out_w = input_size[0] * 2, input_size[1] * 2

        # Move to device (GPU)
        blurry_frames, depths, blur_maps = (
            blurry_frames.unsqueeze(0).to(device),
            depths.unsqueeze(0).to(device),
            blur_maps.unsqueeze(0).to(device),
        )

        binary_masks = dilate_mask((blur_maps[0] > 0.1).float()).unsqueeze(0)

        # print("Blurry frames shape: ", blurry_frames.shape, "/n(Min, Max) = (", blurry_frames.min().item(), ", " ,blurry_frames.max().item(), ")")
        # print("Depths shape: ", depths.shape, "/n(Min, Max) = (", depths.min().item(), ", " ,depths.max().item(), ")")
        # print("blur_maps shape: ", blur_maps.shape, "/n(Min, Max) = (", blur_maps.min().item(), ", " ,blur_maps.max().item(), ")")

        video_length = blurry_frames.shape[1]

        pbar.update(1)
        pbar.set_description(f"Processing: {video_name} ({video_length} frames)")
        # print(f"Processing: {video_name} ({video_length} frames)")

        ##############################################
        # Flow Completion
        ##############################################

        with torch.no_grad():
            # ---- compute flow ----
            if blurry_frames.size(-1) <= 640:
                short_clip_len = 12
            elif blurry_frames.size(-1) <= 720:
                short_clip_len = 8
            elif blurry_frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            # use fp32 for RAFT
            if video_length > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = raft(
                            blurry_frames[:, f:end_f], iters=args.raft_iter
                        )
                    else:
                        flows_f, flows_b = raft(
                            blurry_frames[:, f - 1 : end_f], iters=args.raft_iter
                        )

                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()

                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = raft(blurry_frames, iters=args.raft_iter)
                torch.cuda.empty_cache()

            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > args.subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, args.subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + args.subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                    pred_flows_bi_sub, _ = flow_completion.forward_bidirect_flow(
                        blurry_frames[:, s_f : e_f + 1],
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        binary_masks[:, s_f : e_f + 1],
                    )
                    pred_flows_bi_sub = flow_completion.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        pred_flows_bi_sub,
                        binary_masks[:, s_f : e_f + 1],
                    )

                    pred_flows_f.append(
                        pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e]
                    )
                    pred_flows_b.append(
                        pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e]
                    )
                    torch.cuda.empty_cache()

                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                pred_flows_bi, _ = flow_completion.forward_bidirect_flow(
                    blurry_frames, gt_flows_bi, binary_masks
                )
                pred_flows_bi = flow_completion.combine_flow(
                    gt_flows_bi, pred_flows_bi, binary_masks
                )
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
                        pred_flows_bi[0][:, s_f : e_f - 1],
                        pred_flows_bi[1][:, s_f : e_f - 1],
                    )
                    prop_imgs_sub, updated_local_masks_sub = (
                        depthpainter.img_propagation(
                            blurry_frames[:, s_f:e_f],
                            pred_flows_bi_sub,
                            binary_masks[:, s_f:e_f],
                            "nearest",
                        )
                    )
                    updated_frames_sub = (
                        blurry_frames[:, s_f:e_f] * (1 - binary_masks[:, s_f:e_f])
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
                prop_imgs, updated_local_masks = depthpainter.img_propagation(
                    blurry_frames, pred_flows_bi, binary_masks, interpolation="nearest"
                )
                updated_frames = (
                    blurry_frames * (1 - binary_masks)
                    + prop_imgs.view(b, t, 3, h, w) * binary_masks
                )
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
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

        for f in tqdm(range(0, video_length, neighbor_stride), leave=False):
            neighbor_ids = [
                i
                for i in range(
                    max(0, f - neighbor_stride),
                    min(video_length, f + neighbor_stride + 1),
                )
            ]
            ref_ids = get_ref_index(
                f, neighbor_ids, video_length, args.ref_stride, ref_num
            )
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_blur_maps = blur_maps[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
            )
            selected_depths = depths[:, neighbor_ids + ref_ids, :, :, :]

            with torch.no_grad():
                l_t = len(neighbor_ids)
                ori_pred_img = depthpainter(
                    selected_imgs,
                    selected_depths,
                    selected_pred_flows_bi,
                    selected_blur_maps,
                    selected_update_masks,
                    l_t,
                )
                pred_img = ori_pred_img.view(-1, 3, out_h, out_w).permute(0, 2, 3, 1)
                pred_img = (pred_img - torch.min(pred_img)) / (
                    torch.max(pred_img) - torch.min(pred_img)
                )
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i].cpu().numpy() * 255).astype(np.uint8)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = (
                            comp_frames[idx].astype(np.float32) * 0.5
                            + img.astype(np.float32) * 0.5
                        )

                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)
                    # comp_frames[idx] = img.astype(np.uint8)

            torch.cuda.empty_cache()

        # save videos
        if args.save_videos:
            # Create output directory
            save_root = os.path.join(args.output, video_name)
            os.makedirs(save_root, exist_ok=True)

            masked_frame_for_save = []
            green_masked_frame_for_save = []
            for i in range(len(blurry_frames[0])):
                img = np.array(
                    ((blurry_frames[0][i] + 1) / 2).cpu().permute(1, 2, 0) * 255
                )
                masked_frame_for_save.append(img.astype(np.uint8))
                mask_ = np.array(binary_masks[0][i].cpu().permute(1, 2, 0))
                green = np.zeros([h, w, 3])
                green[:, :, 1] = 255
                alpha = 0.6
                fuse_img = (1 - alpha) * img + alpha * green
                fuse_img = mask_ * fuse_img + (1 - mask_) * img
                green_masked_frame_for_save.append(fuse_img.astype(np.uint8))

            imageio.mimwrite(
                os.path.join(save_root, "masked_in.mp4"),
                masked_frame_for_save,
                fps=10,
                quality=7,
            )
            imageio.mimwrite(
                os.path.join(save_root, "green_masked_in.mp4"),
                green_masked_frame_for_save,
                fps=10,
                quality=7,
            )
            imageio.mimwrite(
                os.path.join(save_root, "inpaint_out.mp4"),
                comp_frames,
                fps=10,
                quality=7,
            )

            pbar.update(1)
            pbar.set_description(f"Videos saved.")
            # print(f"Results are saved in {save_root}")
            torch.cuda.empty_cache()

        # compute metrics
        pbar.set_description(f"Computing metrics for {video_name}")

        gt_dir = os.path.join(args.input, "gt_resized")
        gt_frames = get_gt_frames(gt_dir, video_name)

        psnr_list, ssim_list = [], []
        for comp_frame, gt_frame in tqdm(zip(comp_frames, gt_frames), leave=False):

            psnr, ssim = calc_psnr_and_ssim(comp_frame, gt_frame)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

        results_dict["PSNR"].append(np.mean(psnr_list))
        results_dict["SSIM"].append(np.mean(ssim_list))

        pbar.update(1)
        pbar.write(
            f"{video_name} PSNR: {np.mean(psnr_list):.2f}, SSIM: {np.mean(ssim_list):.4f}"
        )

        msg = (
            # f"{video_name} PSNR: {np.mean(psnr_list):.2f}, SSIM: {np.mean(ssim_list):.4f}"
            # + "\n"
            "{:<15s} -- {}".format(
                f"{video_name}",
                f"PSNR: {np.mean(psnr_list):.2f}, SSIM: {np.mean(ssim_list):.4f}",
            )
            + "\n"
        )
        logfile.write(msg)

    msg = (
        "\n"
        + "{:<15s} -- {}".format(
            "Average", {k: round(np.mean(results_dict[k]), 3) for k in metrics}
        )
        + "\n\n"
    )

    print(msg, end="")
    logfile.write(msg)
    logfile.close()
