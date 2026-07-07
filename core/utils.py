import os
import cv2
import random
import numpy as np

import torch
import torchvision.transforms.functional as F
import torch.nn as nn

import pytorch_wavelets as pw


# ###########################################################################
# Frame sampling
# ###########################################################################


def get_ref_index(length, sample_length, local_idx):
    """Pick `sample_length` global reference frames outside the local window,
    spread evenly before/after it in proportion to how many frames lie on each
    side. Used by training, validation and evaluation alike."""

    smaller_idx = sorted([i for i in range(length) if i < min(local_idx)])
    larger_idx = sorted([i for i in range(length) if i > max(local_idx)])

    total_available = len(smaller_idx) + len(larger_idx)
    smaller_samples = round(sample_length * len(smaller_idx) / total_available)
    larger_samples = sample_length - smaller_samples

    ref_idx = []
    if smaller_samples > 0:
        smaller_step = len(smaller_idx) // smaller_samples
        start_idx = smaller_step // 2
        ref_idx.extend(
            [smaller_idx[start_idx + i * smaller_step] for i in range(smaller_samples)]
        )

    if larger_samples > 0:
        larger_step = len(larger_idx) // larger_samples
        start_idx = (larger_step - 1) // 2
        ref_idx.extend(
            [larger_idx[start_idx + i * larger_step] for i in range(larger_samples)]
        )

    return sorted(ref_idx)


# ###########################################################################
# Create random blur masks
# ###########################################################################


def get_random_focus_depths():

    # Define focal range
    window = int(random.uniform(5, 10)) * 10
    focal_point = random.uniform(0, 255)

    # Add focus pull
    dfdt = random.randint(0, 5)

    return window, dfdt, focal_point


def get_bk_amount(depth_value, min_blur, max_blur, focus_range, focal_point):

    focus_upper = focal_point + focus_range // 2
    focus_lower = focal_point - focus_range // 2

    if depth_value < focus_lower:
        blur = int(max_blur - ((depth_value / focus_lower) * (max_blur - min_blur)))
    elif depth_value > focus_upper:
        blur = int(
            min_blur
            + (
                ((depth_value - focus_upper) / (255 - focus_upper))
                * (max_blur - min_blur)
            )
        )
    else:
        blur = 1

    if blur % 2 == 0:
        blur += 1
    return blur


def gaussian_blur(input_tensor, bk, sigma):
    """Gaussian blur via the functional API (no per-call module construction)."""
    return F.gaussian_blur(input_tensor, kernel_size=bk, sigma=float(sigma)).int()


DEPTH_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npz", ".npy", ".pfm", ".exr")


def load_depth(depth_dir, frame_name, frame_idx, depth_files):
    """Load a depth map for a frame, accepting any common on-disk format.

    Resolution order:
      1. name-based: ``<base>_depth.{png,npz,npy}`` then ``<base>.{png,npz,npy}``
      2. index-based: ``depth_files[frame_idx]`` (e.g. YouTube-VOS ``00000.jpg``
         <-> ``depth_0000.npz``, aligned by sorted order).

    Returns a float32 numpy array (H, W) normalised to [0, 255], in the repo's
    convention (closer = brighter, matching get_depths.py). 8-bit image depths
    are kept as-is; raw arrays (npz/npy) and >8-bit images are min-max scaled,
    and raw arrays are inverted to match the image-depth convention.
    """
    base = os.path.splitext(frame_name)[0]
    path = None
    for cand in (
        base + "_depth.png", base + "_depth.npz", base + "_depth.npy",
        base + ".png", base + ".npz", base + ".npy",
    ):
        p = os.path.join(depth_dir, cand)
        if os.path.exists(p):
            path = p
            break
    if path is None:  # fall back to index alignment (ignore stray/marker files)
        valid = [f for f in depth_files if os.path.splitext(f)[1].lower() in DEPTH_EXTS]
        if not valid:
            raise FileNotFoundError(f"No depth files in {depth_dir}")
        path = os.path.join(depth_dir, valid[min(frame_idx, len(valid) - 1)])

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data.files else data.files[0]
        depth = np.asarray(data[key]).astype(np.float32)
        raw = True
    elif ext == ".npy":
        depth = np.load(path).astype(np.float32)
        raw = True
    else:  # png / jpg / tiff, 8- or 16-bit
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read depth: {path}")
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth = depth.astype(np.float32)
        raw = False

    depth = np.squeeze(depth)

    # Normalise raw arrays and any >8-bit images to [0, 255]; leave canonical
    # 8-bit image depths untouched so existing datasets are unchanged.
    if raw or depth.max() > 255:
        dmin, dmax = float(depth.min()), float(depth.max())
        depth = (depth - dmin) / (dmax - dmin + 1e-8) * 255.0
    if raw:
        depth = 255.0 - depth  # closer = brighter, matching get_depths.py

    return depth


def get_wavelet(img):

    img = nn.functional.interpolate(
        img.unsqueeze(0), size=(img.shape[-2] * 2, img.shape[-1] * 2), mode="bilinear"
    )
    transform = pw.DWTForward(J=1, wave="haar", mode="zero").to(img.device)
    yL, yh = transform(img)
    b, c, wt, h, w = yh[0].shape
    yh = yh[0].view(wt, b, c, h, w)

    wavelet = torch.abs(yh[0]) + torch.abs(yh[1]) + torch.abs(yh[2])
    wavelet = torch.sum(wavelet, 1, keepdim=True)

    return wavelet[0]


def get_blur_map(img, depth):

    wavelet = get_wavelet(img) / 255
    depth = depth / 255

    rounded_depth = torch.round(depth, decimals=1)
    depth_values = torch.unique(rounded_depth[wavelet > 0])

    blur_map = torch.zeros_like(depth).to(depth.device)
    for d in depth_values:

        wavelet_sum = torch.sum(wavelet[rounded_depth == d])

        blur_map[rounded_depth == d] = wavelet_sum

    output = 1 - (blur_map / (torch.max(blur_map) + 1e-8))

    return output


def blur_with_depth(
    img,
    depth,
    min_blur=0,
    max_blur=13,
    focal_point=100,
    focus_range=100,
    sigma=5,
    num_layers=100,
):
    """
    Apply depth-based blurring to an image.
    This function blurs an image based on the depth map provided. The amount of blur
    is determined by the depth value at each pixel, with closer pixels receiving less
    blur and farther pixels receiving more blur.

    Parameters:
        img (torch.Tensor): The input image tensor, shape [3, H, W], normalized to [0, 255].
        depth (torch.Tensor): The depth map tensor, shape [H, W], normalized to [0, 255].
        min_blur (int, optional): The minimum blur amount. Default is 1.
        max_blur (int, optional): The maximum blur amount. Default is 13.
        focal_point (float, optional): The focal point in the depth map, normalized to [0, 255]. Default is 100.
        focus_range (int, optional): The range of depth values around the focal point that remain in focus, 0-254. Default is 100.
        sigma (int, optional): The standard deviation for the Gaussian blur. Default is 5.
        num_layers (int, optional): The number of depth layers to process. Default is 100.

    Returns:
        torch.Tensor: The blurred image tensor, sclaed between [0, 255].

    """

    out = torch.zeros(img.shape).to(img.device)
    step = 255 / num_layers
    assert step > 1, "Depth map and img should be normalized to [0, 255]"

    # Each depth slab uses one Gaussian kernel size; many slabs share a size, so
    # blur the frame once per distinct size and reuse it (same result, far fewer
    # convolutions than blurring once per layer).
    blur_cache = {}

    for depth_value in torch.arange(0, 255, step):
        mask = torch.zeros(depth.shape).to(depth.device)
        if depth_value == 0:
            mask[depth == 0] = 1
        mask[depth > depth_value] = 1
        mask[depth > (depth_value + step)] = 0

        blur_amount = get_bk_amount(
            depth_value, min_blur, max_blur, focus_range, focal_point
        )
        if blur_amount == 1:
            masked_img = img * mask
        else:
            if blur_amount not in blur_cache:
                blur_cache[blur_amount] = gaussian_blur(img, blur_amount, sigma)
            masked_img = blur_cache[blur_amount] * mask

        out = torch.add(out, masked_img)

    blurred_img = (out - out.min()) / (out.max() - out.min() + 1e-8) * 255
    return blurred_img
