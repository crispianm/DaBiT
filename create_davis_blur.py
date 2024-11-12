import os
import glob
import torch
from tqdm import tqdm

from core.utils import get_blur_map, get_wavelet, blur_with_depth
import cv2

out_dir = "./blur_tests"
davis_path = "/home/wg19671/Documents/davis/frames"
davis_depth_path = "/home/wg19671/Documents/davis/depths"
height = 240
width = 432


def create_davis_blur_test(davis_path, davis_depth_path, out_dir):

    for folder_name in tqdm(sorted(os.listdir(davis_path))):
        folder_path = os.path.join(davis_path, folder_name)
        depth_path = os.path.join(davis_depth_path, folder_name)

        if not os.path.exists(os.path.join(out_dir, folder_name)):
            print(f"Creating folder {os.path.join(out_dir, folder_name)}")
            os.makedirs(os.path.join(out_dir, folder_name))

        mask_dir = os.path.join(out_dir, folder_name, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        frame_out_dir = os.path.join(out_dir, folder_name, "frames")
        os.makedirs(frame_out_dir, exist_ok=True)
        depth_out_dir = os.path.join(out_dir, folder_name, "depths")
        os.makedirs(depth_out_dir, exist_ok=True)

        image_files = sorted(glob.glob(os.path.join(folder_path, "*")))
        depth_files = sorted(glob.glob(os.path.join(depth_path, "*")))
        num_frames = len(image_files)

        step = 255 / (num_frames)
        max_blur = 13
        window_size = 100
        sigma = 5
        focal_point = 0

        for frame_idx in tqdm(range(num_frames), leave=False):

            # if os.path.exists(os.path.join(mask_dir, f"{frame_idx:05}_mask.png")):
            #     continue

            img = cv2.imread(image_files[frame_idx])
            img = cv2.resize(img, (width, height))

            depth = cv2.imread(depth_files[frame_idx], cv2.IMREAD_GRAYSCALE)
            depth = cv2.resize(depth, (width, height))

            img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() 
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
            focal_point += step
        
            cv2.imwrite(os.path.join(frame_out_dir, f"{frame_idx:05}.png"), cv2.cvtColor(blurred_img.permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(depth_out_dir, f"{frame_idx:05}_depth.png"), depth.permute(1,2,0).numpy())
            cv2.imwrite(os.path.join(mask_dir, f"{frame_idx:05}_mask.png"), blur_mask.permute(1,2,0).numpy() * 255)

        break


if __name__ == "__main__":
    create_davis_blur_test(davis_path, davis_depth_path, out_dir)
