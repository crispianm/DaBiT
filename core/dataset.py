import os
import random

import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
import torchvision.io as io

from model.modules.depth_anything_v2.dpt import DepthAnythingV2

from core.utils import (
    GroupRandomHorizontalFlip,
    blur_with_depth,
    get_random_focus_depths,
    get_blur_map,
)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, dataset: dict, device):
        self.args = args
        self.video_root = dataset["video_root"]
        self.device = device

        self.num_local_frames = args["num_local_frames"]
        self.num_ref_frames = args["num_ref_frames"]
        self.ori_size = self.ori_w, self.ori_h = (args["w"], args["h"])
        self.w, self.h = self.ori_w // 2, self.ori_h // 2

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

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        encoder = "vitl"  # or 'vits', 'vitb', 'vitg'

        depth_model = DepthAnythingV2(**model_configs[encoder])
        depth_model.load_state_dict(
            torch.load(f"./weights/depth_anything_v2_{encoder}.pth", map_location="cpu", weights_only=True)
        )
        self.depth_model = depth_model.to(self.device).eval()

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
        frames = []
        blurred_frames = []
        masks = []

        for idx in selected_local_index:

            # Load GT Image and resized image
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.ori_w, self.ori_h))
            frames.append(torch.tensor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)).permute(2, 0, 1))
            img = cv2.resize(img, (self.w, self.h))

            # Load Depth Image and resize
            depth = self.depth_model.infer_image(img)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = 255 - depth

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

            if len(frames) == self.num_local_frames:  # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    blurred_frames.reverse()
                    masks.reverse()

        (frames,blurred_frames,masks) = GroupRandomHorizontalFlip()(frames, blurred_frames, masks)

        # Normalize to Tensors
        gt_tensors = torch.stack(frames) / 255.0
        input_tensors = torch.stack(blurred_frames) / 255.0
        mask_tensors = torch.stack(masks)


        return (
            gt_tensors,
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
