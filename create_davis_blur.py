import os
import glob
import torch
from tqdm import tqdm
import torchvision.io as io
import torchvision.transforms as transforms

from core.utils import get_blur_map, get_wavelet, blur_with_depth

out_dir = "T:/ProPainter Datasets/davis/blur_tests"
davis_path = "T:/ProPainter Datasets/davis/gt"
davis_depth_path = "T:/ProPainter Datasets/davis/depths"
height = 480
width = 864


def create_davis_blur_test(davis_path, davis_depth_path, out_dir):

    for folder_name in tqdm(os.listdir(davis_path)):
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

        image_files = glob.glob(os.path.join(folder_path, "*"))
        depth_files = glob.glob(os.path.join(depth_path, "*"))
        num_frames = len(image_files)

        step = 255 / (num_frames)
        max_blur = 7
        window_size = 100
        focal_point = 0

        for frame_idx in tqdm(range(num_frames), leave=False):

            if os.path.exists(os.path.join(mask_dir, f"{frame_idx:05}_mask.png")):
                continue

            img = io.read_image(image_files[frame_idx]) / 255.0
            depth = io.read_image(depth_files[frame_idx])[0].unsqueeze(0) / 255.0

            transform = transforms.Resize((height, width))
            img = transform(img)
            depth = transform(depth)

            blurred_img = blur_with_depth(
                img * 255,
                depth[0] * 255,
                0,
                max_blur,
                focal_point,
                window_size,
                sigma=5,
                num_layers=100,
            )
            blurred_wavelet = get_wavelet(blurred_img)
            blur_mask = get_blur_map(blurred_wavelet, depth)
            focal_point += step

            io.write_png(
                (blurred_img * 255).byte(),
                os.path.join(frame_out_dir, f"{frame_idx:05}.png"),
            )
            io.write_png(
                (depth * 255).byte(),
                os.path.join(depth_out_dir, f"{frame_idx:05}_depth.png"),
            )
            io.write_png(
                (blur_mask * 255).to(torch.uint8),
                os.path.join(mask_dir, f"{frame_idx:05}_mask.png"),
            )

            # plt.imsave(os.path.join(frame_out_dir, f"{frame_idx:05}.png"), blurred_img.permute(1,2,0).numpy())
            # plt.imsave(os.path.join(depth_out_dir, f"{frame_idx:05}_depth.png"), depth.permute(1,2,0).numpy(), cmap='gray')
            # plt.imsave(os.path.join(mask_dir, f"{frame_idx:05}_mask.png"), blur_mask.permute(1,2,0).numpy(), cmap='gray')


if __name__ == "__main__":
    create_davis_blur_test(davis_path, davis_depth_path, out_dir)
