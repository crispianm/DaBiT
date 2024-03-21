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

# matplotlib.use('agg')

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


def to_tensors():
    return transforms.Compose([Stack(), ToTorchFormatTensor()])


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
        flowF_group,
        flowB_group,
    ):
        v = random.random()
        if v < 0.5:
            ret_img = [F.hflip(img) for img in img_group]
            ret_blurred_img = [
                F.hflip(blurred_img) for blurred_img in blurred_img_group
            ]
            ret_depth = [F.hflip(depth) for depth in depth_group]
            ret_mask = [F.hflip(mask) for mask in mask_group]
            ret_flowF = [ff[:, ::-1] * [-1.0, 1.0] for ff in flowF_group]
            ret_flowB = [fb[:, ::-1] * [-1.0, 1.0] for fb in flowB_group]
            return ret_img, ret_blurred_img, ret_depth, ret_mask, ret_flowF, ret_flowB
        else:
            return (
                img_group,
                blurred_img_group,
                depth_group,
                mask_group,
                flowF_group,
                flowB_group,
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


class ToTorchFormatTensor(object):
    """Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]"""

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


# ###########################################################################
# Create masks with random shape
# ###########################################################################


def create_random_shape_with_random_motion(
    video_length, imageHeight=240, imageWidth=432
):
    # get a random shape
    height = random.randint(imageHeight // 3, imageHeight - 1)
    width = random.randint(imageWidth // 3, imageWidth - 1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8) / 10

    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width
    )
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(0, imageHeight - region_height), random.randint(
        0, imageWidth - region_width
    )
    v = get_random_v(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
    masks = [m.convert("L")]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks * video_length
    # return moving masks
    for _ in range(video_length - 1):
        x, y, v = random_move_control_points(
            x,
            y,
            imageHeight,
            imageWidth,
            v,
            region.size,
            maxLineAcceleration=(3, 0.5),
            maxInitSpeed=3,
        )
        m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
        masks.append(m.convert("L"))
    return masks


def create_random_shape_with_random_motion_zoom_rotation(
    video_length,
    zoomin=0.9,
    zoomout=1.1,
    rotmin=1,
    rotmax=10,
    imageHeight=240,
    imageWidth=432,
):
    # get a random shape
    assert zoomin < 1, "Zoom-in parameter must be smaller than 1"
    assert zoomout > 1, "Zoom-out parameter must be larger than 1"
    assert (
        rotmin < rotmax
    ), "Minimum value of rotation must be smaller than maximun value !"
    height = random.randint(imageHeight // 3, imageHeight - 1)
    width = random.randint(imageWidth // 3, imageWidth - 1)
    edge_num = random.randint(6, 8)
    ratio = random.randint(6, 8) / 10
    region = get_random_shape(
        edge_num=edge_num, ratio=ratio, height=height, width=width
    )
    region_width, region_height = region.size
    # get random position
    x, y = random.randint(0, imageHeight - region_height), random.randint(
        0, imageWidth - region_width
    )
    v = get_random_v(max_speed=3)
    m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
    m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
    masks = [m.convert("L")]
    # return fixed masks
    if random.uniform(0, 1) > 0.5:
        return masks * video_length  # -> directly copy all the base masks
    # return moving masks
    for _ in range(video_length - 1):
        x, y, v = random_move_control_points(
            x,
            y,
            imageHeight,
            imageWidth,
            v,
            region.size,
            maxLineAcceleration=(3, 0.5),
            maxInitSpeed=3,
        )
        m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        ### add by kaidong, to simulate zoon-in, zoom-out and rotation
        extra_transform = random.uniform(0, 1)
        # zoom in and zoom out
        if extra_transform > 0.75:
            resize_coefficient = random.uniform(zoomin, zoomout)
            region = region.resize(
                (
                    math.ceil(region_width * resize_coefficient),
                    math.ceil(region_height * resize_coefficient),
                ),
                Image.NEAREST,
            )
            m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
            region_width, region_height = region.size
        # rotation
        elif extra_transform > 0.5:
            m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
            m = m.rotate(random.randint(rotmin, rotmax))
            # region_width, region_height = region.size
        ### end
        else:
            m.paste(region, (y, x, y + region.size[0], x + region.size[1]))
        masks.append(m.convert("L"))
    return masks


def get_random_shape(edge_num=9, ratio=0.7, width=432, height=240):
    """
    There is the initial point and 3 points per cubic bezier curve.
    Thus, the curve will only pass though n points, which will be the sharp edges.
    The other 2 modify the shape of the bezier curve.
    edge_num, Number of possibly sharp edges
    points_num, number of points in the Path
    ratio, (0, 1) magnitude of the perturbation from the unit circle,
    """
    points_num = edge_num * 3 + 1
    angles = np.linspace(0, 2 * np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instead of Path.CLOSEPOLY avoids an innecessary straight line
    verts = (
        np.stack((np.cos(angles), np.sin(angles))).T
        * (2 * ratio * np.random.random(points_num) + 1 - ratio)[:, None]
    )
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor="black", lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts) * 1.1, np.max(verts) * 1.1)
    ax.set_ylim(np.min(verts) * 1.1, np.max(verts) * 1.1)
    ax.axis("off")  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8)) * 255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = (
        np.min(corrdinates[0]),
        np.max(corrdinates[0]),
        np.min(corrdinates[1]),
        np.max(corrdinates[1]),
    )
    region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    return region


def random_accelerate(v, maxAcceleration, dist="uniform"):
    speed, angle = v
    d_speed, d_angle = maxAcceleration
    if dist == "uniform":
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == "guassian":
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(f"Distribution type {dist} is not supported.")
    return (speed, angle)


def get_random_v(max_speed=3, dist="uniform"):
    if dist == "uniform":
        speed = np.random.uniform(max_speed)
    elif dist == "guassian":
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(f"Distribution type {dist} is not supported.")
    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def random_move_control_points(
    X,
    Y,
    imageHeight,
    imageWidth,
    linev,
    region_size,
    maxLineAcceleration=(3, 0.5),
    maxInitSpeed=3,
):
    region_width, region_height = region_size
    speed, angle = linev
    X += int(speed * np.cos(angle))
    Y += int(speed * np.sin(angle))
    linev = random_accelerate(linev, maxLineAcceleration, dist="guassian")
    if (
        (X > imageHeight - region_height)
        or (X < 0)
        or (Y > imageWidth - region_width)
        or (Y < 0)
    ):
        linev = get_random_v(maxInitSpeed, dist="guassian")
    new_X = np.clip(X, 0, imageHeight - region_height)
    new_Y = np.clip(Y, 0, imageWidth - region_width)
    return new_X, new_Y, linev


if __name__ == "__main__":

    trials = 10
    for _ in range(trials):
        video_length = 10
        # The returned masks are either stationary (50%) or moving (50%)
        masks = create_random_shape_with_random_motion(
            video_length, imageHeight=240, imageWidth=432
        )

        for m in masks:
            cv2.imshow("mask", np.array(m))
            cv2.waitKey(500)


# ###########################################################################
# Create random blur masks
# ###########################################################################


def get_random_focus_depths():
    """
    Generate random focus depths for depth inpainting.

    Returns:
        tuple: A tuple containing the following values:
            - d1 (float): The lower depth limit.
            - d2 (float): The upper depth limit.
            - v (float): The focus pull value.
            - focal_point (float): The randomly generated focal point.
    """
    # Define focal range
    window = int(random.uniform(6, 12)) * 10
    focal_point = random.uniform(0, 255)

    # Add focus pull
    step = random.uniform(2, 10)

    return window, step, focal_point


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


def blur_with_depth(
    img,
    depth_map,
    min_blur,
    max_blur,
    focal_point,
    focus_range=100,
    sigma=5,
    num_layers=100,
):

    out = torch.zeros(img.shape)
    blur_mask = torch.zeros(depth_map.shape)

    min_depth = torch.min(depth_map)
    max_depth = torch.max(depth_map)
    step = (max_depth - min_depth) / num_layers
    assert step > 1, "Depth map or img should be normalized to [0, 255]"
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
        blur_mask = torch.add(blur_mask, mask * blur_amount)

    return out, blur_mask.unsqueeze(0)


def get_blurred_masked_frames(frame_tensors, mask_tensors):
    """
    Apply blurring to frames based on masks and return the frames blurred where the mask is > 0.

    Args:
        frame_tensors (list): List of frame tensors.
        mask_tensors (list): List of mask tensors.

    Returns:
        torch.Tensor: Tensor containing the masked frames.
    """
    masked_frames = torch.zeros_like(frame_tensors)

    for i in range(len(frame_tensors)):

        bk = random.choice([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23])
        sigma = random.uniform(0.1, 5.0)

        for j in range(len(frame_tensors[i])):
            frame = frame_tensors[i][j]
            mask = mask_tensors[i][j]

            blurred_frame = get_blurred_frame(frame, bk, sigma)

            # Set nonzeros of mask to blurred image
            masked_frames[i][j] = torch.where(mask > 0, blurred_frame, frame)

    return masked_frames
