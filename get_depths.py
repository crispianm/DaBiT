import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


"""
Example Usage:
python get_depths.py -i "./data_in" -o "./data_out"

"""

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_depth_estimate(filepath, image_processor, depth_model, device):
    # Transform input for midas
    image = Image.open(filepath)
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    return depth


def process_folder(
    input_folder_path, output_folder_path, transform, depth_model, device
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
                    depth_estimate = get_depth_estimate(
                        input_file_path, transform, depth_model, device
                    )
                except Exception as e:
                    print(f"Error processing {input_file_path}: {e}")
                    with open("error_log.txt", "a") as f:
                        f.write(f"Error processing {input_file_path}: {e}\n")
                    continue

                # print(f"Saving image {output_file_path}.")
                depth_estimate.save(output_file_path)

        elif item.is_dir():
            # Recursively process subfolder
            subfolder_output_path = os.path.join(output_folder_path, item.name)
            process_folder(
                item.path,
                subfolder_output_path,
                transform,
                depth_model,
                device,
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
        # use apple silicon if no cuda gpu available
        print("No GPU found, using cpu instead")
        device = torch.device("cpu")

    # Load model
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    ).to(device)

    # Define transforms for model
    image_processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )

    process_folder(
        args.input, args.output, image_processor, depth_model, args.device
    )
