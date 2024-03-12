import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


import matplotlib.pyplot as plt


"""
Example Usage:
python get_depths.py -i "./data_in" -o "./data_out"

"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_depth_estimate(filepath, transform, depth_model, device, color):
    # Transform input for midas

    raw_image = cv2.imread(filepath)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    h, w = image.shape[:2]

    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = depth_model(image)

    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[
        0, 0
    ]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    output = depth.cpu().numpy().astype(np.uint8)
    color_output = cv2.applyColorMap(output, cv2.COLORMAP_INFERNO)

    return output, color_output


def process_folder(
    input_folder_path, output_folder_path, transform, depth_model, device, color
):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    for item in os.scandir(input_folder_path):
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
                    depth_estimate, depth_estimate_color = get_depth_estimate(
                        input_file_path, transform, depth_model, device, color=color
                    )
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")
                    with open("error_log.txt", "a") as f:
                        f.write(f"Error processing {input_file_path}: {e}\n")
                    continue

                print(f"Saving image {output_file_path}.")
                if color == True:
                    plt.imsave(output_file_path, depth_estimate_color, cmap="plasma")
                else:
                    plt.imsave(output_file_path, depth_estimate, cmap="binary")

        elif item.is_dir():
            # Recursively process subfolder
            subfolder_output_path = os.path.join(output_folder_path, item.name)
            process_folder(
                item.path,
                subfolder_output_path,
                transform,
                depth_model,
                device,
                color=color,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply MiDaS to a directory of images."
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
        "-c",
        "--color",
        type=bool,
        default=False,
        # choices=[True, False],
        help="Whether to output color images. (Default: False)",
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
        device = torch.device("cpu")
        print("No GPU found, using CPU instead")

    # Load model
    depth_model = (
        DepthAnything.from_pretrained(
            "LiheYoung/depth_anything_{}14".format(args.encoder)
        )
        .to(device)
        .eval()
    )
    total_params = sum(param.numel() for param in depth_model.parameters())
    print("Total parameters: {:.2f}M".format(total_params / 1e6))

    # Define transforms for model
    transform = Compose(
        [
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    process_folder(
        args.input, args.output, transform, depth_model, args.device, args.color
    )
