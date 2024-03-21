import os
import json
import random

import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.io as io

from utils.flow_util import resize_flow, flowread
from core.utils import (
    GroupRandomHorizontalDepthFlip,
    blur_with_depth,
    get_random_focus_depths,
)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, dataset: dict):
        self.args = args
        self.depth_root = dataset["depth_root"]
        self.video_root = dataset["video_root"]
        self.flow_root = dataset["flow_root"]

        self.num_local_frames = args["num_local_frames"]
        self.num_ref_frames = args["num_ref_frames"]
        self.ori_size = self.ori_w, self.ori_h = (args["w"], args["h"])
        self.w, self.h = self.ori_w // 2, self.ori_h // 2

        self.load_flow = args["load_flow"]
        if self.load_flow:
            assert os.path.exists(self.flow_root)

        self.load_depth = args["load_depth"]
        if self.load_depth:
            assert os.path.exists(self.depth_root)

        if dataset["name"] == "youtube-vos":

            json_path = os.path.join(
                self.video_root.split("youtube-vos")[0], "youtube-vos/train.json"
            )
            with open(json_path, "r") as f:
                self.video_train_dict = json.load(f)
            self.video_names = sorted(list(self.video_train_dict.keys()))
        else:
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

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]

        if self.args["use_blur_masks"]:
            window, step, focal_point = get_random_focus_depths()
            max_blur = random.randint(3, 13)

        # create sample index
        selected_index = self._sample_index(
            self.video_dict[video_name], self.num_local_frames, self.num_ref_frames
        )

        # read video frames
        frames = []
        blurred_frames = []
        masks = []
        depths = []
        flows_f, flows_b = [], []

        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img = io.read_image(img_path)
            img = transforms.Resize(size=(self.ori_h, self.ori_w), antialias=None)(img)
            frames.append(img)
            img = transforms.Resize(size=(self.h, self.w), antialias=None)(img)
            # print(img.shape)

            if self.load_depth:
                depth_path = os.path.join(
                    self.depth_root, video_name, frame_list[idx][:-4] + "_depth.png"
                )
                depth = io.read_image(depth_path)[0].unsqueeze(0)
                depth = transforms.Resize(size=(self.h, self.w), antialias=None)(depth)
                depths.append(depth)
            else:
                raise ValueError("Depth not loaded")

            if self.args["use_blur_masks"]:
                blurred_image, blur_mask = blur_with_depth(
                    img,
                    depth[0],
                    0,
                    max_blur,
                    focal_point,
                    100,
                    sigma=5,
                    num_layers=10,
                )
                focal_point += step

                blurred_frames.append(blurred_image)
                masks.append(blur_mask / torch.max(blur_mask))

            if len(frames) <= self.num_local_frames - 1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx + 1][:-4]
                flow_f_path = os.path.join(
                    self.flow_root, video_name, f"{current_n}_{next_n}_f.flo"
                )
                flow_b_path = os.path.join(
                    self.flow_root, video_name, f"{next_n}_{current_n}_b.flo"
                )
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if len(frames) == self.num_local_frames:  # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    blurred_frames.reverse()
                    depths.reverse()
                    masks.reverse()
                    if self.load_flow:
                        flows_f.reverse()
                        flows_b.reverse()
                        flows_ = flows_f
                        flows_f = flows_b
                        flows_b = flows_

        (
            frames,
            blurred_frames,
            depths,
            masks,
            flows_f,
            flows_b,
        ) = GroupRandomHorizontalDepthFlip()(
            frames, blurred_frames, depths, masks, flows_f, flows_b
        )

        # normalize to tensors
        frame_tensors = (torch.stack(frames) / 255.0 * 2.0) - 1.0
        blurred_frame_tensors = (torch.stack(blurred_frames) / 255.0 * 2.0) - 1.0
        depth_tensors = torch.stack(depths) / 255.0
        mask_tensors = torch.stack(masks)

        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1)  # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        # img [-1,1] mask [0,1]
        if self.load_flow:
            return (
                frame_tensors,
                blurred_frame_tensors,
                depth_tensors,
                mask_tensors,
                flows_f,
                flows_b,
                video_name,
            )
        else:
            return (
                frame_tensors,
                blurred_frame_tensors,
                depth_tensors,
                mask_tensors,
                "None",
                "None",
                video_name,
            )


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
