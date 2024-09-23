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


class TrainZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(TrainZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = TrainZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, "r")
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = TrainZipReader.build_file_dict(path)
        filelist = zfile.namelist()
        filelist.sort()
        data = zfile.read(filelist[idx])
        #
        im = Image.open(io.BytesIO(data))
        return im


class TestZipReader(object):
    file_dict = dict()

    def __init__(self):
        super(TestZipReader, self).__init__()

    @staticmethod
    def build_file_dict(path):
        file_dict = TestZipReader.file_dict
        if path in file_dict:
            return file_dict[path]
        else:
            file_handle = zipfile.ZipFile(path, "r")
            file_dict[path] = file_handle
            return file_dict[path]

    @staticmethod
    def imread(path, idx):
        zfile = TestZipReader.build_file_dict(path)
        filelist = zfile.namelist()
        filelist.sort()
        data = zfile.read(filelist[idx])
        file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # im = Image.open(io.BytesIO(data))
        return im


# ###########################################################################
# Data augmentation
# ###########################################################################


# def to_tensors():
#     return transforms.Compose([Stack(), ToTorchFormatTensor()])


class GroupRandomHorizontalFlowFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, img_group, mask_group, flowF_group, flowB_group):
        v = random.random()
        if v < 0.5:
            ret_img = [F.hflip(img) for img in img_group]
            ret_mask = [F.hflip(mask) for mask in mask_group]
            ret_flowF = [ff[:, ::-1] * [-1.0, 1.0] for ff in flowF_group]
            ret_flowB = [fb[:, ::-1] * [-1.0, 1.0] for fb in flowB_group]
            return ret_img, ret_mask, ret_flowF, ret_flowB
        else:
            return img_group, mask_group, flowF_group, flowB_group


class GroupRandomHorizontalDepthFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __call__(
        self,
        img_group,
        blurred_img_group,
        depth_group,
        mask_group,
    ):
        v = random.random()
        if v < 0.5:
            ret_img = [F.hflip(img) for img in img_group]
            ret_blurred_img = [
                F.hflip(blurred_img) for blurred_img in blurred_img_group
            ]
            ret_depth = [F.hflip(depth) for depth in depth_group]
            ret_mask = [F.hflip(mask) for mask in mask_group]
            return ret_img, ret_blurred_img, ret_depth, ret_mask
        else:
            return (
                img_group,
                blurred_img_group,
                depth_group,
                mask_group,
            )


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, img_group, mask_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [F.hflip(img) for img in img_group]
            ret_mask = [F.hflip(mask) for mask in mask_group]
            if is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret, ret_mask
        else:
            return img_group, mask_group


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == "1":
            img_group = [img.convert("L") for img in img_group]
            mode = "L"
        if mode == "L":
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == "RGB":
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


# class ToTorchFormatTensor(object):
#     """Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
#     to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]"""

#     def __init__(self, div=True):
#         self.div = div

#     def __call__(self, pic):
#         if isinstance(pic, np.ndarray):
#             # numpy img: [L, C, H, W]
#             img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
#         else:
#             # handle PIL Image
#             img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
#             img = img.view(pic.size[1], pic.size[0], len(pic.mode))
#             # put it from HWC to CHW format
#             # yikes, this transpose takes 80% of the loading time/CPU
#             img = img.transpose(0, 1).transpose(0, 2).contiguous()
#         img = img.float().div(255) if self.div else img.float()
#         return img


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


# def generate_random_depth_mask(depth, d1, d2, v, focal_point):
#     """
#     Generate a random depth mask based on the given depth map and parameters.

#     Args:
#         depth (numpy.ndarray): The depth map.
#         d1 (float): The lower bound of the focal range.
#         d2 (float): The upper bound of the focal range.
#         v (float): The amount of focus pull.
#         focal_point (float): The focal point.

#     Returns:
#         tuple: A tuple containing the updated lower bound of the focal range, the updated upper bound of the focal range,
#                and the generated depth mask as a PIL Image object.
#     """
#     depth = depth / np.max(depth)

#     mask = np.zeros((depth.shape[0], depth.shape[1]))

#     # Add focus pull
#     if focal_point >= np.median(depth):
#         d1 += v
#         d2 -= v
#     else:
#         d1 -= v
#         d2 += v

#     # Infill depths outside of focal range
#     mask[depth < d1] = 1
#     mask[depth > d2] = 1

#     return d1, d2, Image.fromarray(mask).convert("L")


def get_bk_amount(depth_value, min_blur, max_blur, focus_range, focal_point):

    focus_upper = focal_point + focus_range // 2
    focus_lower = focal_point - focus_range // 2

    if depth_value < focus_lower:
        blur = int(
            max_blur - (((depth_value - 0) / (focus_lower - 0)) * (max_blur - min_blur))
        )
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
    transfrm = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
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

def get_blur_map(wavelet, depth):
    
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
    depth_map,
    min_blur=1,
    max_blur=7,
    focal_point=0.5,
    focus_range=100,
    sigma=5,
    num_layers=100,
):

    out = torch.zeros(img.shape)

    min_depth = 0
    max_depth = torch.max(depth_map)
    step = (max_depth - min_depth) / num_layers
    assert step > 1, "Depth map and img should be normalized to [0, 255]"
    layers = torch.arange(min_depth, max_depth, step)

    for depth_value in layers:
        mask = torch.zeros(depth_map.shape, dtype=torch.int32)
        if depth_value == 0:
            mask[depth_map == 0] = 1
        mask[depth_map > depth_value] = 1
        mask[depth_map > (depth_value + step)] = 0

        blur_amount = get_bk_amount(
            depth_value, min_blur, max_blur, focus_range, focal_point
        )

        blurred_slice = gaussian_blur(img, blur_amount, sigma)
        masked_img = blurred_slice * mask

        out = torch.add(out, masked_img)

    return out / 255.0
