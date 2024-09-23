import os
import json
import random

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.io as io

from utils.flow_util import resize_flow, flowread
from core.utils import (
    GroupRandomHorizontalDepthFlip,
    blur_with_depth,
    get_random_focus_depths,
    get_blur_map,
    get_wavelet,
    normalize,
)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, dataset: dict):
        self.args = args
        self.depth_root = dataset["depth_root"]
        self.video_root = dataset["video_root"]

        self.num_local_frames = args["num_local_frames"]
        self.num_ref_frames = args["num_ref_frames"]
        self.ori_size = self.ori_w, self.ori_h = (args["w"], args["h"])
        self.w, self.h = self.ori_w // 2, self.ori_h // 2

        self.load_depth = args["load_depth"]
        if self.load_depth:
            assert os.path.exists(self.depth_root)

        # if dataset["name"] == "youtube-vos":

        #     json_path = os.path.join(
        #         self.video_root.split("youtube-vos")[0], "youtube-vos/train.json"
        #     )
        #     with open(json_path, "r") as f:
        #         self.video_train_dict = json.load(f)
        #     self.video_names = sorted(list(self.video_train_dict.keys()))
        # else:
        self.video_names = sorted(os.listdir(self.video_root))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list

        self.video_names = list(self.video_dict.keys())  # update names

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):

        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot : pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx, ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]

        window, step, focal_point = get_random_focus_depths()
        max_blur = random.randint(3, 11)
        sigma = random.randint(1, 5)

        # create sample index
        selected_local_index, selected_ref_index = self._sample_index(
            self.video_dict[video_name], self.num_local_frames, self.num_ref_frames
        )

        # read video frames
        frames = []
        blurred_frames = []
        masks = []
        depths = []

        for idx in selected_local_index:

            # Load GT Image and resized image
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img = io.read_image(img_path) / 255.0
            img = transforms.Resize(size=(self.ori_h, self.ori_w), antialias=None)(img)
            frames.append(img)
            img = transforms.Resize(size=(self.h, self.w), antialias=None)(img)

            # Load Depth Image and resize
            depth_path = os.path.join(
                self.depth_root, video_name, frame_list[idx][:-4] + "_depth.png"
            )
            depth = io.read_image(depth_path)[0].unsqueeze(0) / 255.0
            depth = transforms.Resize(size=(self.h, self.w), antialias=None)(depth)
            depths.append(depth)

            # Create Blurred Image
            blurred_img = blur_with_depth(
                img * 255,
                depth[0] * 255,
                min_blur=1,
                max_blur=max_blur,
                focal_point=focal_point,
                focus_range=window,
                sigma=sigma,
                num_layers=10,
            )
            blurred_frames.append(blurred_img)
            focal_point += step

            # Create Mask
            blurred_wavelet = get_wavelet(blurred_img)
            blur_map = get_blur_map(blurred_wavelet, depth)
            masks.append(blur_map)

            if len(frames) == self.num_local_frames:  # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    blurred_frames.reverse()
                    depths.reverse()
                    masks.reverse()

        (
            frames,
            blurred_frames,
            depths,
            masks,
        ) = GroupRandomHorizontalDepthFlip()(
            frames, blurred_frames, depths, masks
        )

        # Normalize to Tensors
        gt_tensors = torch.stack(frames)
        input_tensors = torch.stack(blurred_frames)
        depth_tensors = torch.stack(depths)
        mask_tensors = torch.stack(masks)
        # mask_tensors = (masks - torch.min(masks)) * (
        #     1.0 / (torch.max(masks) - torch.min(masks))
        # )

        return (
            gt_tensors,
            input_tensors,
            depth_tensors,
            mask_tensors,
            video_name,
        )


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset="T:/ProPainter Datasets/davis/blur_tests"):

        self.dataset = dataset

        self.ori_size = self.ori_w, self.ori_h = (640, 360)
        self.w, self.h = self.ori_w // 2, self.ori_h // 2


    def __getitem__(self, index):

        video_name = self.dataset[index]

        frame_list = sorted(os.listdir(os.path.join(self.video_root, video_name)))
        frame_list = [f for f in frame_list if f.endswith(".jpg")]

        frames = []
        for f in frame_list:
            img_path = os.path.join(self.video_root, video_name, f)
            img = io.read_image(img_path)
            img = transforms.Resize(size=(self.ori_h, self.ori_w), antialias=None)(img)
            frames.append(img)

        frame_tensors = (torch.stack(frames) / 255.0 * 2.0) - 1.0

        return frame_tensors, video_name


class Sampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets=None, iter=False, samples_per_epoch=1000):
        self.datasets = datasets
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [
            0,
        ]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i - 1].__getitem__(index - self.accum[i - 1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0, len(dataset) - 1))

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch
