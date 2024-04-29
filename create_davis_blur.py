import os
import glob
import torch
from tqdm import tqdm
import torchvision.io as io
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

out_dir = "T:/ProPainter Datasets/davis/blur_tests-480p"
davis_path = "T:/ProPainter Datasets/davis/gt_resized"
davis_depth_path = "T:/ProPainter Datasets/davis/depths"
height = 480
width = 864

def get_bk_amount(depth_value, min_blur, max_blur, focus_range, focal_point):

    focus_upper = focal_point + focus_range//2
    focus_lower = focal_point - focus_range//2

    if depth_value < focus_lower:
        blur = int(max_blur - (((depth_value - 0) / (focus_lower - 0)) * (max_blur - min_blur)))
    elif depth_value > focus_upper:
        blur = int(min_blur + (((depth_value - focus_upper) / (255 - focus_upper)) * (max_blur - min_blur)))
    else:
        blur = 1

    if blur % 2 == 0:
        blur += 1
    return blur


def gaussian_blur(input_tensor, bk, sigma):
    '''Gaussian blur generator'''
    blur_transform = transforms.GaussianBlur(kernel_size=bk, sigma=sigma)
    blurred_tensor = blur_transform(input_tensor)
    return blurred_tensor.int()


def blur_with_depth(img, depth_map, min_blur, max_blur, focal_point, focus_range=100, sigma=5, num_layers=100):

    out = torch.zeros(img.shape)
    blur_mask = torch.zeros(depth_map.shape)

    min_depth = 0
    # assert min_depth == 0, "Depth map contains negative values"
    max_depth = torch.max(depth_map)
    assert max_depth <= 255, "Depth map contains values greater than 255"
    step = (max_depth - min_depth) / num_layers
    layers = torch.arange(min_depth, max_depth, step)

    for depth_value in layers:
        mask = torch.zeros(depth_map.shape, dtype=torch.int32)
        if depth_value == 0:
            mask[depth_map == 0] = 1
        mask[depth_map > depth_value] = 1
        mask[depth_map > (depth_value + step)] = 0

        blur_amount = get_bk_amount(depth_value, min_blur, max_blur, focus_range, focal_point)

        blurred_slice = gaussian_blur(
            img,
            blur_amount,
            sigma
        )
        masked_img = blurred_slice * mask

        out = torch.add(out, masked_img)
        blur_mask = torch.add(blur_mask, mask*blur_amount)

    return out, blur_mask


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
        
        image_files = glob.glob(os.path.join(folder_path, '*'))
        depth_files = glob.glob(os.path.join(depth_path, '*'))
        num_frames = len(image_files)
        
        step = 255/(num_frames)
        max_blur = 5
        window_size = 40
        focal_point = 0

        for frame_idx in tqdm(range(num_frames), leave=False):

            if os.path.exists(os.path.join(mask_dir, f"{frame_idx:05}_mask.png")):
                continue

            img = io.read_image(image_files[frame_idx])
            depth = io.read_image(depth_files[frame_idx])[0]

            transform = transforms.Resize((height, width))
            img = transform(img)
            depth = transform(depth)

            blurred_image, blur_mask = blur_with_depth(img, depth, 0, max_blur, focal_point, window_size/2, sigma=5, num_layers=100)
            focal_point += step

            plt.imsave(os.path.join(mask_dir, f"{frame_idx:05}_mask.png"), blur_mask, cmap='gray')
            plt.imsave(os.path.join(depth_out_dir, f"{frame_idx:05}_depth.png"), depth, cmap='gray')
            plt.imsave(os.path.join(frame_out_dir, f"{frame_idx:05}.png"), blurred_image.permute(1,2,0).numpy()/255.0)
    
if __name__ == "__main__":
    create_davis_blur_test(davis_path, davis_depth_path, out_dir)
    