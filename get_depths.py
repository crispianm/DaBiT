import argparse
import os
import cv2
import torch
from tqdm import tqdm

from model.modules.depth_anything_v2.dpt import DepthAnythingV2


"""
Example Usage:
python get_depths.py -i "./data_in" -o "./data_out"

"""

def process_folder(
    input_folder_path, output_folder_path, depth_model, device
):

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for item in tqdm(os.scandir(input_folder_path), leave=False):
        if item.is_file():
            # Process file
            input_file_path = item.path
            output_file_path = os.path.join(
                output_folder_path, os.path.splitext(item.name)[0] + "_depth.png"
            )
            if os.path.exists(output_file_path):
                print(f"{output_file_path} already exists, skipping.")
            else:
                try:
                    image = cv2.imread(input_file_path)
                    depth = depth_model.infer_image(image)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    depth = 255 - depth
                    cv2.imwrite(output_file_path, depth)
                
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")
                    with open("error_log.txt", "a") as f:
                        f.write(f"Error processing {input_file_path}: {e}\n")
                    continue

        elif item.is_dir():
            # Recursively process subfolder
            subfolder_output_path = os.path.join(output_folder_path, item.name)
            process_folder(
                item.path,
                subfolder_output_path,
                depth_model,
                device,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply DepthAnythingV2 to a directory of images."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to folder containing images to apply MiDaS to. Ex: './data'",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output files to. Ex: './data_out'",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="\nDevice to run the models. (Default: 'cuda')",
    )

    parser.add_argument(
        "-e", "--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"]
    )

    args = parser.parse_args()

    # Check if GPU is available
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
    depth_model.load_state_dict(torch.load(f'./weights/depth_anything_v2_{args.encoder}.pth', map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device).eval()

    process_folder(args.input, args.output, depth_model, device)
