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
    
    
    def _ref_index(self, length, sample_length, local_idx):

        smaller_idx = sorted([i for i in range(length) if i < min(local_idx)])
        larger_idx = sorted([i for i in range(length) if i > max(local_idx)])

        total_available = len(smaller_idx) + len(larger_idx)
        smaller_samples = round(sample_length * len(smaller_idx) / total_available)
        larger_samples = sample_length - smaller_samples

        smaller_step = len(smaller_idx) // smaller_samples if smaller_samples > 0 else 0
        larger_step = len(larger_idx) // larger_samples if larger_samples > 0 else 0

        ref_idx = []
        if smaller_samples > 0:
            smaller_step = len(smaller_idx) // smaller_samples
            start_idx = smaller_step // 2 
            ref_idx.extend([smaller_idx[start_idx + i * smaller_step] for i in range(smaller_samples)])

        if larger_samples > 0:
            larger_step = len(larger_idx) // larger_samples
            start_idx = (larger_step - 1) // 2 
            ref_idx.extend([larger_idx[start_idx + i * larger_step] for i in range(larger_samples)])

        return sorted(ref_idx)
    

    def __getitem__(self, index):

        video_name = self.video_names[index]

        window, dfdt, focal_point = get_random_focus_depths()
        max_blur = random.randint(3, 13)
        sigma = random.randint(2, 5)

        # create sample index
        selected_local_index = self._sample_index(
            self.video_dict[video_name], self.num_local_frames
        )
        selected_ref_index = self._ref_index(
            self.video_dict[video_name], self.num_ref_frames, selected_local_index
        )
        selected_index = sorted(selected_local_index + selected_ref_index)
        local_index = [i for i in range(len(selected_index)) if selected_index[i] in selected_local_index]

        # read video frames
        gt_frames = []
        resized_frames = []
        blurred_frames = []
        masks = []

        for idx in selected_index:

            # Load GT Image and resized image
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.ori_w, self.ori_h))
            gt_frames.append(torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).to(self.device))
            img = cv2.resize(img, (self.w, self.h))
            resized_frames.append(torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).to(self.device))

            # Load Depth Image and resize
            depth_path = os.path.join(self.depth_root, video_name, frame_list[idx][:-4] + "_depth.png")
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            gt_depth = cv2.resize(gt_depth, (self.w, self.h))

            # Convert numpy arrays to tensors
            img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).float().to(self.device)
            gt_depth = torch.tensor(gt_depth).float().to(self.device)

            if idx in selected_local_index:
                # Create Blurred Image
                blurred_img = blur_with_depth(
                    img,
                    gt_depth,
                    min_blur=0,
                    max_blur=max_blur,
                    focal_point=focal_point,
                    focus_range=window,
                    sigma=sigma,
                    num_layers=30,
                )
                blurred_frames.append(blurred_img)
                focal_point += dfdt

                # Create Mask
                blur_map = get_blur_map(blurred_img, gt_depth.unsqueeze(0))
                masks.append(blur_map)

            elif idx in selected_ref_index:

                blurred_img = blur_with_depth(
                    img,
                    gt_depth,
                    min_blur=0,
                    max_blur=random.randint(3, 13),
                    focal_point=random.uniform(0, 255),
                    focus_range=int(random.uniform(5, 10)) * 10,
                    sigma=random.randint(2, 5),
                    num_layers=30,
                )
                blurred_frames.append(blurred_img)

                # Create Mask
                blur_map = get_blur_map(blurred_img, gt_depth.unsqueeze(0))
                masks.append(blur_map)

            else:
                raise ValueError("Index not found in local or ref index")
                

            
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

        return gt_tensors,resized_tensors,input_tensors,mask_tensors,local_index


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, device, dataset):
        self.video_root = dataset
        self.device = device

        self.ori_w, self.ori_h = (864, 480)
        self.w, self.h = self.ori_w // 2, self.ori_h // 2

        self.video_names = sorted(os.listdir(self.video_root))

        self.video_dict = {}
        self.frame_dict = {}


    def __len__(self):
        return len(self.video_names)


    def __getitem__(self, index):

        video_name = self.video_names[index]

        # Load data
        gt_frames = sorted(os.listdir(os.path.join(self.video_root, video_name, "gt")))
        input_frames = sorted(os.listdir(os.path.join(self.video_root, video_name, "frames")))
        masks = sorted(os.listdir(os.path.join(self.video_root, video_name, "masks")))

        # Read and process frames
        gt_frames_processed = []
        input_frames_processed = []
        maps_processed = []

        for frame in gt_frames:
            img = cv2.imread(os.path.join(self.video_root, video_name, "gt", frame))
            img = cv2.resize(img, (self.ori_w, self.ori_h))
            img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).to(self.device)
            gt_frames_processed.append(img)

        for frame in input_frames:
            img = cv2.imread(os.path.join(self.video_root, video_name, "frames", frame))
            img = cv2.resize(img, (self.w, self.h))
            img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1).to(self.device)
            input_frames_processed.append(img)

        for mask in masks:
            img = cv2.imread(os.path.join(self.video_root, video_name, "masks", mask), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.w, self.h))
            img = torch.tensor(img).unsqueeze(0).to(self.device)
            maps_processed.append(img)

        # Normalize to Tensors
        gt_tensors = torch.stack(gt_frames_processed) / 255.0
        input_tensors = torch.stack(input_frames_processed) / 255.0
        map_tensors = torch.stack(maps_processed) / 255.0

        return gt_tensors,input_tensors,map_tensors,video_name
        


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
