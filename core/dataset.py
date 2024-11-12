import os
import random

import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
import torchvision.io as io

import torchvision.transforms.functional as F

from core.utils import (
    blur_with_depth,
    get_random_focus_depths,
    get_blur_map,
)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, dataset: dict, device):
        self.args = args
        self.video_root = os.path.join(dataset["video_root"], "frames")
        self.depth_root = os.path.join(dataset["video_root"], "depths")
        self.device = device

        self.num_local_frames = args["num_local_frames"]
        self.num_ref_frames = args["num_ref_frames"]
        self.ori_size = self.ori_w, self.ori_h = (args["w_train"], args["h_train"])
        self.w, self.h = self.ori_w // 2, self.ori_h // 2

        self.video_names = sorted(os.listdir(self.video_root))

        self.video_dict = {}
        self.frame_dict = {}

        for name in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, name)))
            length = len(frame_list)
            if length > self.num_local_frames + self.num_ref_frames:
                self.video_dict[name] = length
                self.frame_dict[name] = frame_list
        self.video_names = list(self.video_dict.keys()) 

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length):

        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot : pivot + sample_length]

        return local_idx

    def __getitem__(self, index):

        video_name = self.video_names[index]

        window, step, focal_point = get_random_focus_depths()
        max_blur = random.randint(3, 11)
        sigma = random.randint(1, 5)

        # create sample index
        selected_local_index = self._sample_index(
            self.video_dict[video_name], self.num_local_frames
        )

        # read video frames
        gt_frames = []
        resized_frames = []
        blurred_frames = []
        masks = []

        for idx in selected_local_index:

            # Load GT Image and resized image
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.ori_w, self.ori_h))
            gt_frames.append(torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1))
            img = cv2.resize(img, (self.w, self.h))
            resized_frames.append(torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1))

            # Load Depth Image and resize
            depth_path = os.path.join(self.depth_root, video_name, frame_list[idx][:-4] + "_depth.png")
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth = cv2.resize(depth, (self.w, self.h))


            # Convert numpy arrays to tensors
            img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float() 
            depth = torch.tensor(depth).float()

            # Create Blurred Image
            blurred_img = blur_with_depth(
                img,
                depth,
                min_blur=1,
                max_blur=max_blur,
                focal_point=focal_point,
                focus_range=window,
                sigma=sigma,
                num_layers=100,
            )
            blurred_frames.append(blurred_img)
            focal_point += step

            # Create Mask
            blur_map = get_blur_map(blurred_img, depth.unsqueeze(0))
            masks.append(blur_map)

            if len(gt_frames) == self.num_local_frames:  # random reverse
                if random.random() < 0.5:
                    gt_frames.reverse()
                    resized_frames.reverse()
                    blurred_frames.reverse()
                    masks.reverse()

        # Random Horizontal Flip
        v = random.random()
        if v < 0.5:
            gt_frames = [F.hflip(img) for img in gt_frames]
            resized_frames = [F.hflip(img) for img in resized_frames]
            blurred_frames = [
                F.hflip(blurred_img) for blurred_img in blurred_frames
            ]
            masks = [F.hflip(mask) for mask in masks]


        # Normalize to Tensors
        gt_tensors = torch.stack(gt_frames) / 255.0
        resized_tensors = torch.stack(resized_frames) / 255.0
        input_tensors = torch.stack(blurred_frames) / 255.0
        mask_tensors = torch.stack(masks)

        return (
            gt_tensors,
            resized_tensors,
            input_tensors,
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
