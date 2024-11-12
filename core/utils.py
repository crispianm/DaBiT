import os
import io
import cv2
import random
import numpy as np
from PIL import Image, ImageOps
import zipfile
import math

import torch
import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.nn as nn

import pytorch_wavelets as pw


# ###########################################################################
# Directory IO
# ###########################################################################


def read_dirnames_under_root(root_dir):
    dirnames = [
        name
        for i, name in enumerate(sorted(os.listdir(root_dir)))
        if os.path.isdir(os.path.join(root_dir, name))
    ]
    print(f"Reading directories under {root_dir}, num: {len(dirnames)}")
    return dirnames


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
    """Gaussian blur generator"""
    blur_transform = transforms.GaussianBlur(kernel_size=bk, sigma=sigma)
    blurred_tensor = blur_transform(input_tensor)
    return blurred_tensor.int()


def normalize(x):
    if x.dtype != torch.float32:
        x = x.float()
    transfrm = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transfrm(x)


def get_wavelet(img):

    img = nn.functional.interpolate(
        img.unsqueeze(0), size=(img.shape[-2] * 2, img.shape[-1] * 2), mode="bilinear"
    )
    transform = pw.DWTForward(J=1, wave="haar", mode="zero")

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

    blur_map = torch.zeros_like(depth)
    for d in depth_values:

        wavelet_sum = torch.sum(wavelet[rounded_depth == d])

        blur_map[rounded_depth == d] = wavelet_sum

    output = 1 - (blur_map / torch.max(blur_map))

    return output


def blur_with_depth(
    img,
    depth,
    min_blur=1,
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

    out = torch.zeros(img.shape)
    step = 255 / num_layers
    assert step > 1, "Depth map and img should be normalized to [0, 255]"

    for depth_value in torch.arange(0, 255, step):
        mask = torch.zeros(depth.shape)
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
            blurred_slice = gaussian_blur(img, blur_amount, sigma)
            masked_img = blurred_slice * mask

        out = torch.add(out, masked_img)

    blurred_img = (out - out.min()) / (out.max() - out.min()) * 255
    return blurred_img
