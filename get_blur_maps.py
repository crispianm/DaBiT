import os
import numpy as np
from tqdm import tqdm 
import pytorch_wavelets as pw
import torch
import torchvision.io as io
import torch.nn.functional as F
import matplotlib.pyplot as plt


def print_summary(name, namestring):
    print(
        f"{namestring} shape: ",
        name.shape,
        "\n\t(Min, Max) = (",
        name.min().item(),
        ", ",
        name.max().item(),
        ")",
        "\n\tUnique values: ",
        torch.unique((name*255).to(torch.uint8)),
        
    )



def norm_tensor(tensor):
    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))

def normalize_blur_maps(blur_map, max_level=11):

    bm_norm = norm_tensor(blur_map) * 255

    norms = [round(255*i/max_level) for i in range(1, max_level+1, 2)]
    blur_map_norm = torch.zeros_like(blur_map)

    for i in range(len(norms)):
        if i == 0:
            blur_map_norm[bm_norm <= norms[i]] = norms[i]
        else:
            blur_map_norm[(bm_norm > norms[i-1]) & (bm_norm <= norms[i])] = norms[i]

    return blur_map_norm

def get_wavelet(img):

    img = F.interpolate(img.unsqueeze(0), size=(img.shape[-2]*2, img.shape[-1]*2), mode='bilinear')
    transform = pw.DWTForward(J=1, wave='haar', mode='zero')

    yL, yh = transform(img)
    b, c, wt, h, w = yh[0].shape
    yh = yh[0].view(wt, b, c, h, w)

    wavelet = torch.abs(yh[0]) + torch.abs(yh[1]) + torch.abs(yh[2])
    wavelet = torch.sum(wavelet, 1, keepdim=True) / 255

    return wavelet[0]

def get_blur_map(image_path, depth_path, levels=9):

    # Read image and depth
    depth = io.read_image(depth_path, mode=io.ImageReadMode.GRAY).float() / 255
    d_l = (levels + 1) / 2
    depth = torch.round(depth * d_l) / d_l
    img_tensor = io.read_image(image_path, mode=io.ImageReadMode.GRAY).float()


    # Get wavelet transform
    wavelet = get_wavelet(img_tensor)
    # wavelet = norm_tensor(wavelet)

    blur_map = torch.zeros_like(depth)
    for d in torch.unique(depth):
        wavelet_sum = torch.sum(wavelet[depth == d])
        wavelet_sum /= torch.sum(depth == d)
        blur_map[depth == d] = 1 - wavelet_sum

    blur_map_norm = normalize_blur_maps(blur_map).to(torch.uint8)
    # blur_map_norm = (blur_map_norm + torch.min(blur_map_norm[blur_map_norm > 0])) / (torch.max(blur_map_norm) + torch.min(blur_map_norm[blur_map_norm > 0])) * 255
    # blur_map_norm[blur_map_norm == 0] = torch.min(blur_map_norm[blur_map_norm > 0])
    
    # return blur_map_norm.to(torch.uint8)
    return blur_map_norm


def process_folder(folder_path):

    for video_name in tqdm(os.listdir(folder_path)):

        out_dir = os.path.join(folder_path, video_name, "wavelet_blur_maps")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        frames_path = os.path.join(folder_path, video_name, "frames")
        depths_path = os.path.join(folder_path, video_name, "depths")

        for img_path, depth_path in tqdm(zip(os.listdir(frames_path), os.listdir(depths_path)), leave=False):
            img_path = os.path.join(frames_path, img_path)
            depth_path = os.path.join(depths_path, depth_path)
            blur_map = get_blur_map(img_path, depth_path)
            blur_map = torch.cat((blur_map, blur_map, blur_map), 0)
            
            blur_map = blur_map.permute(1,2,0).numpy()
            blur_map = ((blur_map / np.max(blur_map)) * 255).astype(np.uint8)


            out_map_path = img_path.replace("frames", "wavelet_blur_maps")
            plt.imsave(out_map_path, blur_map, cmap='gray')
      

process_folder("T:/ProPainter Datasets/davis/blur_tests/")
