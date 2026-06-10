import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch

from model.modules.depth_anything_v2.dpt import DepthAnythingV2
from core.utils import get_blur_map


"""
Example usage:
python get_blur_maps.py -i "./data_in" -o "./data_out"
"""


def process_folder(input_folder, output_folder, depth_model):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in tqdm(sorted(os.listdir(input_folder)), leave=False):

        # Load Image
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        # Estimate depth (closer = brighter, matching get_depths.py)
        depth = depth_model.infer_image(img)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = 255 - depth

        # Convert numpy arrays to tensors
        img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        depth = torch.tensor(depth).float()

        # Create Blur Map
        blur_map = get_blur_map(img, depth.unsqueeze(0))

        # Save Blur Map as PNG
        blur_map = blur_map.squeeze().cpu().numpy()
        blur_map = (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min()) * 255.0
        cv2.imwrite(os.path.join(output_folder, img_name), blur_map.astype(np.uint8))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate per-frame focal-blur maps for a folder of images."
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Folder of input frames.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Folder to write blur maps to.")
    parser.add_argument("-e", "--encoder", type=str, default="vitl",
                        choices=["vits", "vitb", "vitl"],
                        help="Depth Anything V2 encoder (default: vitl).")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using ", torch.cuda.get_device_name(0))
    else:
        print("No GPU found, using cpu instead")
        device = torch.device("cpu")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    depth_model = DepthAnythingV2(**model_configs[args.encoder])
    depth_model.load_state_dict(torch.load(
        f'./weights/depth_anything_v2_{args.encoder}.pth',
        map_location='cpu', weights_only=True,
    ))
    depth_model = depth_model.to(device).eval()

    process_folder(args.input, args.output, depth_model)
