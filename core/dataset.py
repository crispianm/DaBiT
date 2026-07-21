import os
import random

import cv2

import torch
import torchvision.transforms.functional as F

from core.utils import (
    blur_with_depth,
    get_random_focus_depths,
    get_blur_map,
    get_ref_index,
    load_depth,
)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, dataset: dict):
        self.args = args
        # Frames/depths default to <video_root>/frames and <video_root>/depths,
        # but can be overridden per dataset (e.g. YouTube-VOS keeps depths in a
        # separate tree) via frames_root/depths_root or frames_subdir/depths_subdir.
        video_root = dataset["video_root"]
        self.video_root = dataset.get(
            "frames_root",
            os.path.join(video_root, dataset.get("frames_subdir", "frames")),
        )
        self.depth_root = dataset.get(
            "depths_root",
            os.path.join(video_root, dataset.get("depths_subdir", "depths")),
        )

        self.num_local_frames = args["num_local_frames"]
        self.num_ref_frames = args["num_ref_frames"]
        self.ori_size = self.ori_w, self.ori_h = (args["w_train"], args["h_train"])
        self.w, self.h = self.ori_w // 2, self.ori_h // 2

        self.video_names = sorted(os.listdir(self.video_root))

        self.video_dict = {}
        self.frame_dict = {}
        self.depth_dict = {}

        for name in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, name)))
            length = len(frame_list)
            if length > self.num_local_frames + self.num_ref_frames:
                depth_path = os.path.join(self.depth_root, name)
                depth_files = (
                    sorted(os.listdir(depth_path)) if os.path.isdir(depth_path) else []
                )
                if not depth_files:
                    continue  # no depths for this video; skip rather than crash mid-training
                self.video_dict[name] = length
                self.frame_dict[name] = frame_list
                self.depth_dict[name] = depth_files
        self.video_names = list(self.video_dict.keys())
        print(
            f"  {dataset.get('name', '?')}: {len(self.video_names)} sequences"
            f" (frames: {self.video_root})"
        )

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length):

        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot : pivot + sample_length]

        return local_idx
    
    
    def _ref_index(self, length, sample_length, local_idx):
        return get_ref_index(length, sample_length, local_idx)
    

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

            # Load GT Image and resized image (kept on CPU so DataLoader workers
            # can build samples in parallel; the trainer moves them to the GPU)
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.ori_w, self.ori_h))
            gt_frames.append(torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))
            img = cv2.resize(img, (self.w, self.h))
            resized_frames.append(torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))

            # Load Depth (any format: png/jpg/16-bit/npz/npy) and resize
            gt_depth = load_depth(
                os.path.join(self.depth_root, video_name),
                frame_list[idx],
                idx,
                self.depth_dict[video_name],
            )
            gt_depth = cv2.resize(gt_depth, (self.w, self.h))

            # Convert numpy arrays to tensors
            img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
            gt_depth = torch.tensor(gt_depth).float()

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
    def __init__(self, dataset):
        self.video_root = dataset

        self.ori_w, self.ori_h = (864, 480)
        self.w, self.h = self.ori_w // 2, self.ori_h // 2

        self.video_names = sorted(os.listdir(self.video_root))

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

        # Tensors are built on CPU so the DataLoader can decode/resize the next
        # video in background worker processes; the caller moves them to the GPU.
        for frame in gt_frames:
            img = cv2.imread(os.path.join(self.video_root, video_name, "gt", frame))
            img = cv2.resize(img, (self.ori_w, self.ori_h))
            img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            gt_frames_processed.append(img)

        for frame in input_frames:
            img = cv2.imread(os.path.join(self.video_root, video_name, "frames", frame))
            img = cv2.resize(img, (self.w, self.h))
            img = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            input_frames_processed.append(img)

        for mask in masks:
            img = cv2.imread(os.path.join(self.video_root, video_name, "masks", mask), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.w, self.h))
            img = torch.tensor(img).unsqueeze(0)
            maps_processed.append(img)

        # Normalize to Tensors
        gt_tensors = torch.stack(gt_frames_processed) / 255.0
        input_tensors = torch.stack(input_frames_processed) / 255.0
        map_tensors = torch.stack(maps_processed) / 255.0

        return gt_tensors,input_tensors,map_tensors,video_name
        


class Sampler(torch.utils.data.Dataset):
    """Randomly samples across datasets, weighted by their lengths."""

    def __init__(self, datasets, samples_per_epoch=1000):
        self.datasets = datasets
        lengths = [len(dataset) for dataset in datasets]
        self.p_datasets = [l / sum(lengths) for l in lengths]
        self.samples_per_epoch = samples_per_epoch

    def __getitem__(self, index):
        # first sample a dataset, then a sequence from it
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        return dataset[random.randint(0, len(dataset) - 1)]

    def __len__(self):
        return self.samples_per_epoch
