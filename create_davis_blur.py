import os
import glob
import argparse
import torch
from tqdm import tqdm

from core.utils import get_blur_map, blur_with_depth
import cv2
import csv

"""
Build the DAVIS-Blur test set from DAVIS-2017 frames + precomputed depths
(see get_depths.py), using the per-sequence blur parameters in davis_blur.csv.

Example usage:
python create_davis_blur.py --frames ./davis/frames --depths ./davis/depths
"""

height = 240
width = 432


def create_davis_blur_test(davis_path, davis_depth_path, out_dir):

    for video_name in tqdm(sorted(os.listdir(davis_path))):
        folder_path = os.path.join(davis_path, video_name)
        depth_path = os.path.join(davis_depth_path, video_name)

        if not os.path.exists(os.path.join(out_dir, video_name)):
            # print(f"Creating folder {os.path.join(out_dir, video_name)}")
            os.makedirs(os.path.join(out_dir, video_name))

        mask_dir = os.path.join(out_dir, video_name, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        frame_out_dir = os.path.join(out_dir, video_name, "frames")
        os.makedirs(frame_out_dir, exist_ok=True)
        depth_out_dir = os.path.join(out_dir, video_name, "depths")
        os.makedirs(depth_out_dir, exist_ok=True)
        gt_dir = os.path.join(out_dir, video_name, "gt")
        os.makedirs(gt_dir, exist_ok=True)

        image_files = sorted(glob.glob(os.path.join(folder_path, "*")))
        depth_files = sorted(glob.glob(os.path.join(depth_path, "*")))
        num_frames = len(image_files)

        # step = 255 / (num_frames)
        # max_blur = 13
        # window_size = 100
        # sigma = 5
        # focal_point = 0
        with open("davis_blur.csv", mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["video_name"] == video_name:
                    max_blur = int(row["max_blur"])
                    window_size = int(row["window_size"])
                    sigma = float(row["sigma"])
                    focal_point = int(row["focal_point"])
                    dfdt = int(row["dfdt"])
                    break

        for frame_idx in tqdm(range(num_frames), leave=False):

            # if os.path.exists(os.path.join(mask_dir, f"{frame_idx:05}_mask.png")):
            #     continue

            img = cv2.imread(image_files[frame_idx])
            gt_img = cv2.resize(img, (width*2, height*2))
            img = cv2.resize(img, (width, height))

            depth = cv2.imread(depth_files[frame_idx], cv2.IMREAD_GRAYSCALE)
            depth = cv2.resize(depth, (width, height))

            img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() 
            depth = torch.tensor(depth).float().unsqueeze(0)


            blurred_img = blur_with_depth(
                img,
                depth,
                min_blur=0,
                max_blur=max_blur,
                focal_point=focal_point,
                focus_range=window_size,
                sigma=sigma,
                num_layers=254,
            )
            # cv2.imwrite(os.path.join(frame_out_dir, f"{frame_idx:05}_wavelet.png"), blurred_wavelet.numpy())
            blur_mask = get_blur_map(blurred_img, depth)
            focal_point += dfdt
        
            cv2.imwrite(os.path.join(gt_dir, f"{frame_idx:05}.png"), cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(frame_out_dir, f"{frame_idx:05}.png"), cv2.cvtColor(blurred_img.permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(mask_dir, f"{frame_idx:05}_mask.png"), blur_mask.permute(1,2,0).numpy() * 255)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the DAVIS-Blur test set from DAVIS frames and depths."
    )
    parser.add_argument("--frames", type=str, required=True,
                        help="DAVIS frames root (one folder per sequence).")
    parser.add_argument("--depths", type=str, required=True,
                        help="DAVIS depths root (from get_depths.py).")
    parser.add_argument("-o", "--out_dir", type=str, default="./blur_tests",
                        help="Output root for the test set (default: ./blur_tests).")
    args = parser.parse_args()

    create_davis_blur_test(args.frames, args.depths, args.out_dir)