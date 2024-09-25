import os
import cv2
import numpy as np
from tqdm import tqdm 
import pytorch_wavelets as pw
import torch
import torchvision.io as io
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.modules.depth_anything_v2.dpt import DepthAnythingV2
from core.utils import get_blur_map


def process_folder(input, output, depth_model, device):

    if not os.path.exists(output):
        os.makedirs(output)

    for img_path in tqdm(sorted(os.listdir(input)), leave=False):

        # Load Image
        img_path = os.path.join(input, img_path)
        img = cv2.imread(img_path)

        # Load Depth Image and resize
        depth = depth_model.infer_image(img)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = 255 - depth

        # Convert numpy arrays to tensors
        img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() 
        depth = torch.tensor(depth).float()
        
        # Create Blur Map
        blur_map = get_blur_map(img, depth.unsqueeze(0))
        # Save Blur Map as PNG
        blur_map = blur_map.squeeze().cpu().numpy()
        blur_map = (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min()) * 255.0
        blur_map = blur_map.astype(np.uint8)
        blur_map_path = os.path.join(output, os.path.basename(img_path))
        cv2.imwrite(blur_map_path, blur_map)
      

if __name__ == "__main__":

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using ", torch.cuda.get_device_name(0))
    else:
        # use apple silicon if no cuda gpu available
        print("No GPU found, using cpu instead")
        device = torch.device("cpu")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'

    depth_model = DepthAnythingV2(**model_configs[encoder])
    depth_model.load_state_dict(torch.load(f'./weights/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device).eval()

    input = "/home/wg19671/Documents/bvidvc/frames/CTaksinBridgeVidevo_960x544_23fps_10bit_420"
    output = "/home/wg19671/Documents/test"

    process_folder(input, output, depth_model, device)
