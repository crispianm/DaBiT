# -*- coding: utf-8 -*-
"""Run one or more DaBiT models on a single (blurred input, sharp GT) clip pair.

Inputs may be video files (mp4/mov/mkv/avi...) or folders of frames. The GT is
used ONLY for metrics: blur maps are estimated from the blurred input itself
(Depth Anything V2 depth of the input frame -> ``core.utils.get_blur_map``, the
same recipe as ``get_blur_maps.py``), so the run is what you would get on a real
clip without ground truth.

Protocol matches ``test_dabit.py`` / DAVIS-Blur: the input is resized to the
inference size (default 432x240), DaBiT restores and 2x super-resolves it, and
metrics are computed against the GT resized to 2x the inference size (864x480).
PSNR/SSIM/tOF come from ``core.metrics`` (identical to the benchmark), LPIPS is
AlexNet. A baseline row scores the blurred input (bicubic-upscaled) the same way.

Example:
    python run_clip.py --input blurred.mp4 --gt sharp.mp4 -o results_clip/myclip
"""
import os
import csv
import json
import math
import time
import argparse
import subprocess

import cv2
import numpy as np
import torch
from torch.nn import functional
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))

# Best released models (see README "Improved models"); local file names.
MODEL_PRESETS = {
    "perceptual": ("dabit_isambard_perceptual.pth", "Perceptual+temporal FT (LPIPS loss)"),
    "retrained": ("dabit_isambard_300k.pth", "Retrained, expanded data"),
    "paper": ("dabit.pth", "Paper (WACVW'25)"),
}
DEFAULT_MODELS = ["perceptual", "retrained", "paper"]
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- io
def read_frames(path, rb_swapped=False):
    """Return (list of RGB uint8 frames, fps or None)."""
    fps = None
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if n.lower().endswith(IMG_EXTS))
        if not names:
            raise SystemExit(f"No image files found in {path}")
        frames = []
        for n in names:
            img = cv2.imread(os.path.join(path, n), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"Could not read {n}")
            frames.append(img)
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        cap.release()
        if not frames:
            raise SystemExit(f"No frames decoded from {path}")
    # cv2 decodes BGR. rb_swapped=True is only for blur_tests/*/gt, whose PNGs
    # were written R/B-swapped by create_davis_blur.py.
    if not rb_swapped:
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    return frames, fps


def write_video(frames_rgb, path, fps, crf=12):
    h, w = frames_rgb[0].shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames_rgb:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def label(img, text):
    img = img.copy()
    scale = img.shape[1] / 864.0
    cv2.rectangle(img, (0, 0), (img.shape[1], int(34 * scale)), (0, 0, 0), -1)
    cv2.putText(img, text, (int(8 * scale), int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7 * scale, (255, 255, 255), max(1, int(2 * scale)), cv2.LINE_AA)
    return img


def grid(tiles, cols):
    h, w = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    out = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    return out


# ---------------------------------------------------------------- blur maps
def estimate_blur_maps(frames_rgb, size, depth_model):
    """Blur maps from the blurred input (get_blur_maps.py / create_davis_blur recipe)."""
    from core.utils import get_blur_map

    w, h = size
    maps = []
    for f in tqdm(frames_rgb, desc="blur maps", leave=False, disable=not os.isatty(2)):
        # Depth Anything on the full-res frame (BGR in, like get_depths.py),
        # 0-255, inverted, stored as uint8 then resized to inference size.
        d = depth_model.infer_image(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255.0
        d = (255 - d).astype(np.uint8)
        d = cv2.resize(d, (w, h))
        small = cv2.resize(f, (w, h))
        img_t = torch.tensor(small).permute(2, 0, 1).float()
        bm = get_blur_map(img_t, torch.tensor(d).float().unsqueeze(0))
        bm = np.clip(bm.squeeze().numpy() * 255, 0, 255).astype(np.uint8)  # PNG round-trip
        maps.append(bm)
    return maps


def load_maps_dir(path, n, size):
    names = sorted(n_ for n_ in os.listdir(path) if n_.lower().endswith(IMG_EXTS))[:n]
    return [cv2.resize(cv2.imread(os.path.join(path, m), cv2.IMREAD_GRAYSCALE), size) for m in names]


# ------------------------------------------------------------------ inference
@torch.no_grad()
def restore(dabit, raft, depth_model, input_tensors, blur_maps, raft_iter=20, amp=None):
    """Verbatim port of the test_dabit.py inference loop for one video.

    input_tensors: [1, T, 3, h, w] RGB in [0, 1] on device
    blur_maps:     [1, T, 1, h, w] in [0, 1] on device
    Returns a list of T RGB uint8 frames at (2h, 2w).
    """
    from core.utils import get_ref_index

    amp = amp or torch.autocast(device_type="cuda", enabled=False)
    h, w = input_tensors.shape[-2:]
    out_h, out_w = 2 * h, 2 * w
    binary_masks = torch.ones_like(blur_maps)
    T = input_tensors.shape[1]

    short_clip_len = 12
    if T > short_clip_len:
        ff, fb = [], []
        for f in range(0, T, short_clip_len):
            e = min(T, f + short_clip_len)
            s = f if f == 0 else f - 1
            a, b = raft(input_tensors[:, s:e], iters=raft_iter)
            ff.append(a)
            fb.append(b)
            torch.cuda.empty_cache()
        flows_bi = (torch.cat(ff, 1), torch.cat(fb, 1))
    else:
        flows_bi = raft(input_tensors, iters=raft_iter)

    sub = 80
    if T > sub:
        upd_f, upd_m = [], []
        pad_len = 10
        for f in range(0, T, sub):
            s_f = max(0, f - pad_len)
            e_f = min(T, f + sub + pad_len)
            pad_s = max(0, f) - s_f
            pad_e = e_f - min(T, f + sub)
            b, t = binary_masks[:, s_f:e_f].shape[:2]
            fl = (flows_bi[0][:, s_f:e_f - 1], flows_bi[1][:, s_f:e_f - 1])
            prop, lm = dabit.img_propagation(input_tensors[:, s_f:e_f], fl,
                                             binary_masks[:, s_f:e_f], "nearest")
            uf = (input_tensors[:, s_f:e_f] * (1 - binary_masks[:, s_f:e_f])
                  + prop.view(b, t, 3, h, w) * binary_masks[:, s_f:e_f])
            um = lm.view(b, t, 1, h, w)
            upd_f.append(uf[:, pad_s:e_f - s_f - pad_e])
            upd_m.append(um[:, pad_s:e_f - s_f - pad_e])
            torch.cuda.empty_cache()
        updated_frames, updated_masks = torch.cat(upd_f, 1), torch.cat(upd_m, 1)
    else:
        b, t = binary_masks.shape[:2]
        prop, lm = dabit.img_propagation(input_tensors, flows_bi, binary_masks, interpolation="nearest")
        updated_frames = input_tensors * (1 - binary_masks) + prop.view(b, t, 3, h, w) * binary_masks
        updated_masks = lm.view(b, t, 1, h, w)

    frame_depths = []
    with amp:
        for s in range(0, T, 12):
            e = min(T, s + 12)
            db = depth_model.preprocess_tensor(updated_frames[0, s:e])
            db = depth_model.forward(db).unsqueeze(1)
            db = functional.interpolate(db, (h, w), mode="bilinear", align_corners=True)
            for k in range(db.shape[0]):
                d = db[k]
                frame_depths.append(1 - ((d - d.min()) / (d.max() - d.min())))
    frame_depths = torch.stack(frame_depths)

    comp = [None] * T
    for f in range(0, T, 5)[:-1]:
        start, end = (T - 10, T) if f + 10 > T else (f, f + 10)
        nids = list(range(start, end))
        rids = get_ref_index(T, 6, nids)
        sel = nids + rids
        imgs = updated_frames[:, sel]
        flows = raft(imgs, iters=raft_iter)
        with amp:
            pred = dabit(imgs, frame_depths[sel].unsqueeze(0), flows,
                         blur_maps[:, sel], updated_masks[:, sel], len(nids))
        pred = pred.float().view(-1, 3, out_h, out_w).permute(0, 2, 3, 1).clamp(0, 1)
        for i, idx in enumerate(nids):
            img = (pred[i].cpu().numpy() * 255).astype(np.uint8)
            if comp[idx] is None:
                comp[idx] = img
            else:
                comp[idx] = (comp[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5).astype(np.uint8)
    return comp


# -------------------------------------------------------------------- metrics
def score(frames_rgb, gt_rgb, lpips_model, device):
    from core.metrics import calc_psnr_and_ssim, calculate_tof

    comp = [cv2.resize(f, (g.shape[1], g.shape[0])).astype(np.float32) for f, g in zip(frames_rgb, gt_rgb)]
    gts = [g.astype(np.float32) for g in gt_rgb]
    psnr, ssim, tof = [], [], [float("nan")]
    for i, (c, g) in enumerate(zip(comp, gts)):
        p, s = calc_psnr_and_ssim(c, g)
        psnr.append(float(p))
        ssim.append(float(s))
        if i > 0:
            tof.append(float(calculate_tof(comp[i - 1], c, gts[i - 1], g)))
    lp = []
    with torch.no_grad():
        for s in range(0, len(comp), 16):
            c = torch.from_numpy(np.stack(comp[s:s + 16]) / 255.0).permute(0, 3, 1, 2).float().to(device)
            g = torch.from_numpy(np.stack(gts[s:s + 16]) / 255.0).permute(0, 3, 1, 2).float().to(device)
            lp.extend(lpips_model(c, g, normalize=True).view(-1).cpu().tolist())
    return {"PSNR": psnr, "SSIM": ssim, "LPIPS": lp, "tOF": tof}


def summarise(per):
    return {k: float(np.nanmean(v)) for k, v in per.items()}


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Blurred clip: video file or frame folder.")
    ap.add_argument("--gt", required=True, help="Sharp ground truth: video file or frame folder.")
    ap.add_argument("-o", "--output", required=True, help="Output folder.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help=f"Preset names {list(MODEL_PRESETS)} or paths to .pth files.")
    ap.add_argument("--weights_dir", default=os.path.join(HERE, "weights"))
    ap.add_argument("--size", default="432x240", help="Inference size WxH (output is 2x). Default 432x240.")
    ap.add_argument("--maps", default=None, help="Optional folder of precomputed blur maps (skips estimation).")
    ap.add_argument("--map_encoder", default="vitl", choices=["vits", "vitb", "vitl"],
                    help="Depth Anything encoder for blur-map depth (default vitl, as get_blur_maps.py).")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None, help="Output video fps (default: input fps or 24).")
    ap.add_argument("--raft_iter", type=int, default=20)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (faster, metrics differ slightly).")
    ap.add_argument("--no_frames", action="store_true", help="Do not write per-frame PNGs.")
    ap.add_argument("--gt_rb_swapped", action="store_true",
                    help="GT frames are stored R/B-swapped (only blur_tests/*/gt from create_davis_blur.py).")
    args = ap.parse_args()

    os.chdir(HERE)  # model code loads relative paths
    import lpips as lpips_lib
    from model.dabit import DaBiT
    from model.modules.flow_comp_raft import RAFT_bi
    from model.modules.depth_anything_v2.dpt import MODEL_CONFIGS, DepthAnythingV2

    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w, h = (int(x) for x in args.size.lower().split("x"))
    if w % 8 or h % 8:
        raise SystemExit("--size must be multiples of 8")
    os.makedirs(args.output, exist_ok=True)

    inp, fps_in = read_frames(args.input)
    gt, fps_gt = read_frames(args.gt, rb_swapped=args.gt_rb_swapped)
    if len(inp) != len(gt):
        print(f"WARNING: input has {len(inp)} frames, GT has {len(gt)}; using the first {min(len(inp), len(gt))}.")
    n = min(len(inp), len(gt), args.max_frames or 10 ** 9)
    inp, gt = inp[:n], gt[:n]
    if n < 10:
        raise SystemExit(f"Need at least 10 frames (DaBiT uses 10-frame windows); got {n}.")
    fps = args.fps or fps_in or fps_gt or 24.0
    ar_in, ar_run = inp[0].shape[1] / inp[0].shape[0], w / h
    if abs(ar_in / ar_run - 1) > 0.05:
        print(f"WARNING: clip aspect {ar_in:.3f} differs from inference aspect {ar_run:.3f}; frames are stretched. "
              f"Consider --size (e.g. {int(round(h * ar_in / 8)) * 8}x{h}).")
    print(f"{n} frames | input {inp[0].shape[1]}x{inp[0].shape[0]} | GT {gt[0].shape[1]}x{gt[0].shape[0]} "
          f"| inference {w}x{h} -> {2 * w}x{2 * h} | fps {fps:g}")

    # Protocol resizes (same cv2 defaults as core.dataset.TestDataset).
    inp_small = [cv2.resize(f, (w, h)) for f in inp]
    gt_eval = [cv2.resize(g, (2 * w, 2 * h)) for g in gt]
    inp_up = [cv2.resize(f, (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC) for f in inp_small]

    # Blur maps (from the blurred input only).
    t0 = time.time()
    if args.maps:
        maps = load_maps_dir(args.maps, n, (w, h))
        map_src = f"precomputed ({args.maps})"
    else:
        dm = DepthAnythingV2(**MODEL_CONFIGS[args.map_encoder])
        dm.load_state_dict(torch.load(os.path.join(args.weights_dir, f"depth_anything_v2_{args.map_encoder}.pth"),
                                      map_location="cpu", weights_only=True))
        dm = dm.to(device).eval()
        maps = estimate_blur_maps(inp, (w, h), dm)
        del dm
        torch.cuda.empty_cache()
        map_src = f"estimated from blurred input (Depth Anything V2 {args.map_encoder})"
    print(f"blur maps: {map_src} [{time.time() - t0:.1f}s]")

    raft = RAFT_bi(model_path=os.path.join(args.weights_dir, "raft.pth"), device=device).to(device)
    depth_model = DepthAnythingV2(**MODEL_CONFIGS["vits"])
    depth_model.load_state_dict(torch.load(os.path.join(args.weights_dir, "depth_anything_v2_vits.pth"),
                                           map_location="cpu", weights_only=True))
    depth_model = depth_model.to(device).eval()
    lpips_model = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp)

    x = torch.from_numpy(np.stack(inp_small)).permute(0, 3, 1, 2).float().div(255).unsqueeze(0).to(device)
    bm = torch.from_numpy(np.stack(maps)).float().div(255).unsqueeze(1).unsqueeze(0).to(device)

    results = {"Input (bicubic x2)": score(inp_up, gt_eval, lpips_model, device)}
    outputs = {}
    for m in args.models:
        if m in MODEL_PRESETS:
            fname, desc = MODEL_PRESETS[m]
            path = os.path.join(args.weights_dir, fname)
            name = m
        else:
            path, name, desc = m, os.path.splitext(os.path.basename(m))[0], m
        net = DaBiT()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net = net.to(device).eval()
        t0 = time.time()
        frames = restore(net, raft, depth_model, x, bm, args.raft_iter, amp)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n
        del net
        torch.cuda.empty_cache()
        key = f"{name}"
        results[key] = score(frames, gt_eval, lpips_model, device)
        outputs[key] = frames
        s = summarise(results[key])
        print(f"{key:12s} [{desc}] PSNR {s['PSNR']:.2f}  SSIM {s['SSIM']:.4f}  LPIPS {s['LPIPS']:.4f}  "
              f"tOF {s['tOF']:.3f}  ({dt:.3f}s/frame)")
        odir = os.path.join(args.output, key)
        os.makedirs(odir, exist_ok=True)
        if not args.no_frames:
            fdir = os.path.join(odir, "frames")
            os.makedirs(fdir, exist_ok=True)
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(fdir, f"{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        write_video(frames, os.path.join(args.output, f"{key}.mp4"), fps)

    # Comparison video: input | models... | GT, labelled, 2 columns x rows at 864x480 tiles.
    names = list(outputs)
    tiles_per = [[label(inp_up[i], "Input (blurred)")]
                 + [label(outputs[k][i], k) for k in names]
                 + [label(gt_eval[i], "Ground truth")] for i in range(n)]
    cols = 3 if len(tiles_per[0]) > 4 else 2
    write_video([grid(t, cols) for t in tiles_per], os.path.join(args.output, "comparison.mp4"), fps, crf=16)

    # Tables.
    summ = {k: summarise(v) for k, v in results.items()}
    best = {"PSNR": max, "SSIM": max, "LPIPS": min, "tOF": min}
    model_keys = [k for k in summ if k in outputs]
    lines = [f"# DaBiT on `{os.path.basename(os.path.normpath(args.input))}`", "",
             f"{n} frames; inference {w}x{h}, scored at {2 * w}x{2 * h} against the GT "
             f"(GT resized from {gt[0].shape[1]}x{gt[0].shape[0]}). Blur maps: {map_src}. "
             "PSNR/SSIM/tOF as test_dabit.py; LPIPS AlexNet. Bold = best model.", "",
             "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | tOF ↓ |", "|---|---:|---:|---:|---:|"]
    for k, s in summ.items():
        cells = []
        for met, fmt in (("PSNR", "{:.2f}"), ("SSIM", "{:.4f}"), ("LPIPS", "{:.4f}"), ("tOF", "{:.3f}")):
            v = fmt.format(s[met])
            if k in model_keys and fmt.format(best[met](summ[mk][met] for mk in model_keys)) == v:
                v = f"**{v}**"
            cells.append(v)
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(args.output, "results.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump({"frames": n, "size": [w, h], "blur_maps": map_src, "summary": summ,
                   "models": {k: (MODEL_PRESETS[k][0] if k in MODEL_PRESETS else k) for k in model_keys}}, f, indent=2)
    with open(os.path.join(args.output, "per_frame.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "method", "PSNR", "SSIM", "LPIPS", "tOF"])
        for k, per in results.items():
            for i in range(n):
                wr.writerow([i, k] + [f"{per[m][i]:.5f}" for m in ("PSNR", "SSIM", "LPIPS", "tOF")])

    try:  # per-frame plot (handy for focus pulls)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
        for k, per in results.items():
            for ax, m in zip(axs, ("PSNR", "SSIM", "LPIPS")):
                ax.plot(per[m], label=k, lw=1.4, ls="--" if k not in outputs else "-")
        for ax, m in zip(axs, ("PSNR ↑", "SSIM ↑", "LPIPS ↓")):
            ax.set_title(m)
            ax.set_xlabel("frame")
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.output, "per_frame.png"), dpi=110)
    except Exception as e:  # plotting is optional
        print(f"(skipped plot: {e})")

    print("\n" + md)
    print(f"Outputs in {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
