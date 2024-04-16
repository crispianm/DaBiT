# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.depthpainter import DepthPainter
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


#  read frames from video
def read_frame_from_videos(root_dir):
    if root_dir.endswith(
        ("mp4", "mov", "avi", "MP4", "MOV", "AVI")
    ):  # input video path
        video_name = os.path.basename(root_dir)[:-4]
        vframes, aframes, info = torchvision.io.read_video(
            filename=root_dir, pts_unit="sec"
        )  # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info["video_fps"]
    else:
        video_name = os.path.basename(root_dir)
        to_gray = transforms.Grayscale()

        frame_path = os.path.join(root_dir, "frames")
        frames = []
        fr_lst = sorted(os.listdir(frame_path))
        for fr in fr_lst:
            frame = torchvision.io.read_image(os.path.join(frame_path, fr))
            frames.append(frame)

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

    return torch.stack(frames), torch.stack(depths), torch.stack(masks), fps, video_name


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
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


if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using ", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU instead")
    device = get_device()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="T:/ProPainter Datasets/davis/blur_tests/bear",
        help="Path of the input video or image folder.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="T:/ProPainter Datasets/davis/blur_tests/bear/results",
        help="Output folder. Default: results",
    )
    parser.add_argument(
        "--ref_stride", type=int, default=5, help="Stride of global reference frames."
    )
    parser.add_argument(
        "--neighbor_length",
        type=int,
        default=5,
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

    args = parser.parse_args()

    frames, depths, masks, fps, video_name = read_frame_from_videos(args.input)
    save_root = os.path.join(args.output, video_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    frames = (frames / 255 * 2) - 1  # norm to -1, 1
    frames = frames[:, :3, :, :]  # only use RGB channels
    depths = depths[:, 0, :, :].unsqueeze(1) / 255.0  # norm to 0, 1
    masks = masks[:, 0, :, :].unsqueeze(1) / 255.0  # norm to 0, 1
    input_size = (frames[0].shape[-1], frames[0].shape[-2])  # width, height
    w, h = input_size
    out_h, out_w = h * 2, w * 2

    # Load the depth model
    # complete depths
    completed_depths = depths

    # Move to device (GPU)
    frames, completed_depths, masks = (
        frames.unsqueeze(0).to(device),
        completed_depths.unsqueeze(0).to(device),
        masks.unsqueeze(0).to(device),
    )

    ##############################################
    # Set up RAFT and flow competition model
    ##############################################
    ckpt_path = load_file_from_url(
        url=os.path.join("raft-things.pth"),
        model_dir="weights",
        progress=True,
        file_name=None,
    )
    fix_raft = RAFT_bi(ckpt_path, device)

    ckpt_path = load_file_from_url(
        url=os.path.join("recurrent_flow_completion.pth"),
        model_dir="weights",
        progress=True,
        file_name=None,
    )
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    ##############################################
    # Set up DepthPainter model
    ##############################################
    ckpt_path = load_file_from_url(
        url=os.path.join("model_320000.pth"),
        model_dir="weights",
        progress=True,
        file_name=None,
    )
    model = DepthPainter(model_path=ckpt_path).to(device)
    model.eval()

    ##############################################
    # DepthPainter inference
    ##############################################
    video_length = frames.shape[1]
    print(f"Processing: {video_name} ({video_length} frames)")
    with torch.no_grad():
        # ---- compute flow ----
        if frames.size(-1) <= 640:
            short_clip_len = 12
        elif frames.size(-1) <= 720:
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        # use fp32 for RAFT
        if video_length > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(
                        frames[:, f:end_f], iters=args.raft_iter
                    )
                else:
                    flows_f, flows_b = fix_raft(
                        frames[:, f - 1 : end_f], iters=args.raft_iter
                    )

                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()

            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=args.raft_iter)
            torch.cuda.empty_cache()

        # if use_half:
        #     frames, flow_masks, masks_dilated = (
        #         frames.half(),
        #         flow_masks.half(),
        #         masks_dilated.half(),
        #     )
        #     gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
        #     fix_flow_complete = fix_flow_complete.half()
        #     model = model.half()

        # # ---- complete flow ----
        # flow_length = gt_flows_bi[0].size(1)
        # if flow_length > args.subvideo_length:
        #     pred_flows_f, pred_flows_b = [], []
        #     pad_len = 5
        #     for f in range(0, flow_length, args.subvideo_length):
        #         s_f = max(0, f - pad_len)
        #         e_f = min(flow_length, f + args.subvideo_length + pad_len)
        #         pad_len_s = max(0, f) - s_f
        #         pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
        #         pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
        #             (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
        #             flow_masks[:, s_f : e_f + 1],
        #         )
        #         pred_flows_bi_sub = fix_flow_complete.combine_flow(
        #             (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
        #             pred_flows_bi_sub,
        #             flow_masks[:, s_f : e_f + 1],
        #         )

        #         pred_flows_f.append(
        #             pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e]
        #         )
        #         pred_flows_b.append(
        #             pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e]
        #         )
        #         torch.cuda.empty_cache()

        #     pred_flows_f = torch.cat(pred_flows_f, dim=1)
        #     pred_flows_b = torch.cat(pred_flows_b, dim=1)
        #     pred_flows_bi = (pred_flows_f, pred_flows_b)
        # else:
        #     pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(
        #         gt_flows_bi, flow_masks
        #     )
        #     pred_flows_bi = fix_flow_complete.combine_flow(
        #         gt_flows_bi, pred_flows_bi, flow_masks
        #     )
        #     torch.cuda.empty_cache()

        pred_flows_bi = gt_flows_bi
        masks_dilated = masks

        # ---- image propagation ----
        masked_frames = frames
        subvideo_length_img_prop = min(
            100, args.subvideo_length
        )  # ensure a maximum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (
                    pred_flows_bi[0][:, s_f : e_f - 1],
                    pred_flows_bi[1][:, s_f : e_f - 1],
                )
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                    masked_frames[:, s_f:e_f],
                    pred_flows_bi_sub,
                    masks_dilated[:, s_f:e_f],
                    "nearest",
                )
                updated_frames_sub = (
                    frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f])
                    + prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
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
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated, interpolation="nearest"
            )
            updated_frames = (
                frames * (1 - masks_dilated)
                + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            )
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()

    comp_frames = [None] * video_length

    neighbor_stride = args.neighbor_length // 2
    if video_length > args.subvideo_length:
        ref_num = args.subvideo_length // args.ref_stride
    else:
        ref_num = -1

    # ---- feature propagation + transformer ----
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i
            for i in range(
                max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1)
            )
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
            pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
        )
        selected_depths = completed_depths[:, neighbor_ids + ref_ids, :, :, :]

        ori_frames = ((frames.permute(0, 1, 3, 4, 2) + 1) / 2 * 255)[0].to(torch.uint8)

        with torch.no_grad():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)

            # pred_img = selected_imgs # results of image propagation
            ori_pred_img = model(
                selected_imgs,
                selected_depths,
                selected_pred_flows_bi,
                selected_masks,
                selected_update_masks,
                l_t,
            )
            pred_img = ori_pred_img.view(-1, 3, out_h, out_w).permute(0, 2, 3, 1)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().numpy() * 255  # print(pred_img.shape)

            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8)
                # print(comp_frames[idx].shape)
                # print(img.shape)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (
                        comp_frames[idx].astype(np.float32) * 0.5
                        + img.astype(np.float32) * 0.5
                    )

                comp_frames[idx] = comp_frames[idx].astype(np.uint8)

        torch.cuda.empty_cache()

    # save videos
    masked_frame_for_save = []
    for i in range(len(frames[0])):
        img = np.array(((frames[0][i] + 1) / 2).cpu().permute(1, 2, 0) * 255)
        masked_frame_for_save.append(img.astype(np.uint8))
    imageio.mimwrite(
        os.path.join(save_root, "masked_in.mp4"),
        masked_frame_for_save,
        fps=10,
        quality=7,
    )
    imageio.mimwrite(
        os.path.join(save_root, "inpaint_out.mp4"), comp_frames, fps=10, quality=7
    )

    print(f"Results are saved in {save_root}")

    torch.cuda.empty_cache()
