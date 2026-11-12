# -*- coding: utf-8 -*-
"""Simulate a focus pull on a sharp clip -> (blurred input video, GT video).

The focal plane sweeps linearly from --fp_start to --fp_end (depth units 0-255,
Depth Anything V2 on the sharp frames, same convention as get_depths.py) across
the clip, using core.utils.blur_with_depth (the DAVIS-Blur / training blur
model). Blur is applied at the output resolution; --max_blur/--sigma are given
in 432x240 units (as in davis_blur.csv) and scaled with the resolution.

Example:
    python make_focus_pull.py --src blur_tests/dog/gt --src_rb_swapped -o sims/dog_pull
    python run_clip.py --input sims/dog_pull/input.mp4 --gt sims/dog_pull/gt.mp4 -o results_clip/dog_pull
"""
import os
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from run_clip import read_frames, write_video, HERE


def write_lossless(frames_rgb, path, fps):
    import subprocess
    h, w = frames_rgb[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps}", "-i", "-", "-c:v", "libx264rgb", "-crf", "0", "-preset", "medium", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames_rgb:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed on {path}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Sharp clip: video file or frame folder.")
    ap.add_argument("--src_rb_swapped", action="store_true", help="Source is blur_tests/*/gt (R/B-swapped PNGs).")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--size", default="864x480", help="Output WxH for both files (default 864x480).")
    ap.add_argument("--fp_start", type=float, default=20.0, help="Focal point at frame 0 (0-255 depth units).")
    ap.add_argument("--fp_end", type=float, default=230.0, help="Focal point at the last frame.")
    ap.add_argument("--window", type=int, default=70, help="In-focus depth range (0-254).")
    ap.add_argument("--max_blur", type=int, default=13, help="Max Gaussian kernel at 432x240.")
    ap.add_argument("--sigma", type=float, default=5.0, help="Gaussian sigma at 432x240.")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--lossy_copy", action="store_true", help="Also write input_h264.mp4 (crf 18, yuv420p).")
    args = ap.parse_args()

    os.chdir(HERE)
    from core.utils import blur_with_depth
    from model.modules.depth_anything_v2.dpt import MODEL_CONFIGS, DepthAnythingV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w, h = (int(x) for x in args.size.lower().split("x"))
    scale = w / 432.0
    max_blur = int(round(args.max_blur * scale)) | 1
    sigma = args.sigma * scale

    frames, fps = read_frames(args.src, rb_swapped=args.src_rb_swapped)
    frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames[: args.max_frames]]
    n = len(frames)

    dm = DepthAnythingV2(**MODEL_CONFIGS["vitl"])
    dm.load_state_dict(torch.load("weights/depth_anything_v2_vitl.pth", map_location="cpu", weights_only=True))
    dm = dm.to(device).eval()

    os.makedirs(args.output, exist_ok=True)
    blurred, fps_log = [], []
    for i, f in enumerate(tqdm(frames, desc="focus pull")):
        with torch.no_grad():
            d = dm.infer_image(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255.0
        d = (255 - d).astype(np.uint8)
        fp = args.fp_start + (args.fp_end - args.fp_start) * i / max(1, n - 1)
        img_t = torch.tensor(f).permute(2, 0, 1).float().to(device)
        out = blur_with_depth(img_t, torch.tensor(d).float().to(device), min_blur=0, max_blur=max_blur,
                              focal_point=fp, focus_range=args.window, sigma=sigma, num_layers=254)
        blurred.append(out.permute(1, 2, 0).clamp(0, 255).round().byte().cpu().numpy())
        fps_log.append(fp)

    write_lossless(blurred, os.path.join(args.output, "input.mp4"), args.fps)
    write_lossless(frames, os.path.join(args.output, "gt.mp4"), args.fps)
    if args.lossy_copy:
        write_video(blurred, os.path.join(args.output, "input_h264.mp4"), args.fps, crf=18)
    with open(os.path.join(args.output, "params.txt"), "w") as fh:
        fh.write(f"src={args.src}\nframes={n}\nsize={w}x{h}\nfp_start={args.fp_start}\nfp_end={args.fp_end}\n"
                 f"window={args.window}\nmax_blur={max_blur} (={args.max_blur}@432)\nsigma={sigma:.2f}\n")
    print(f"{n} frames -> {args.output}/input.mp4, gt.mp4 (focal {args.fp_start:.0f}->{args.fp_end:.0f}, "
          f"max_blur {max_blur}, sigma {sigma:.2f})")


if __name__ == "__main__":
    main()
