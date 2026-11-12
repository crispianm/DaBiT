# -*- coding: utf-8 -*-
"""Side-by-side video for one focal-stack clip: input | models (worst -> best) | focused GT.

The focused GT is an all-in-focus composite of the reference stack: every slice
mean (10 frames) is registered to the middle slice with the chained ECC warps of
eval_focal_stack.py, and each pixel is taken from the slice where its smoothed
Laplacian energy peaks. For each frame the composite is warped back into that
slice's geometry (focus breathing), plus the per-position cross-stack shift for
a low-light stack scored against a lit one (--ref_stack).

Models are ordered by their mean rank over PSNR, SSIM and LPIPS in the
eval_focal_stack.py summary of the same clip (ties broken by LPIPS).
"""
import os
import json
import argparse
import subprocess

import cv2
import numpy as np

from eval_focal_stack import ecc_chain, map_s_to_g

NAMES = {"perceptual": "DaBiT perceptual", "retrained": "DaBiT retrained", "paper": "DaBiT paper"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--ref_stack", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--panel_w", type=int, default=480)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--display_gain", type=float, default=None,
                    help="Brighten input and model panels for display (low light); default: auto when --ref_stack")
    a = ap.parse_args()

    ref_root = a.ref_stack or a.stack
    rmeta = json.load(open(os.path.join(ref_root, "meta.json")))
    emeta = json.load(open(os.path.join(a.stack, "meta.json")))
    F = rmeta["frames_per_slice"]
    rpos = [s["position"] for s in rmeta["slices"]]
    r_idx = {p: i for i, p in enumerate(rpos)}
    epos = [s["position"] for s in emeta["slices"]]

    def mean_at(root, p):
        return sum(cv2.imread(os.path.join(root, f"s{p:05d}_f{f:02d}.png")).astype(np.float32) for f in range(F)) / F

    print("registering + compositing", ref_root, flush=True)
    means = [mean_at(ref_root, p) for p in rpos]
    H, W = means[0].shape[:2]
    small = [cv2.resize(cv2.cvtColor(m, cv2.COLOR_BGR2GRAY), (W // 2, H // 2), interpolation=cv2.INTER_AREA) for m in means]
    A, _ = ecc_chain(small)
    c = len(rpos) // 2
    best = np.full((H, W), -1.0, np.float32)
    comp = np.zeros((H, W, 3), np.float32)
    for s, m in enumerate(means):
        reg = cv2.warpAffine(m, map_s_to_g(A, c, s, 2.0), (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
        g = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
        e = cv2.GaussianBlur(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F) ** 2, (0, 0), 6)
        upd = e > best
        best[upd] = e[upd]
        comp[upd] = reg[upd]
    del means

    def edges(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(cv2.GaussianBlur(np.log1p(g), (0, 0), 2), cv2.CV_32F)

    summ = json.load(open(a.summary))["all"]
    models = [m for m in ("perceptual", "retrained", "paper") if m in summ]
    ranks = {m: 0.0 for m in models}
    for k, higher in (("PSNR", True), ("SSIM", True), ("LPIPS", False)):
        order = sorted(models, key=lambda m: summ[m][k], reverse=higher)
        for r, m in enumerate(order):
            ranks[m] += r
    models = sorted(models, key=lambda m: (-ranks[m], -summ[m]["LPIPS"]))  # worst first
    print("order (worst -> best):", models, flush=True)

    lowlight = a.ref_stack is not None
    gain = a.display_gain
    if lowlight and gain is None:
        x = cv2.imread(os.path.join(a.clip, "00064.png")).astype(np.float32).mean()
        gain = float(comp.mean() / max(x, 1e-3))
    gain = gain or 1.0

    pw = a.panel_w
    ph = int(round(pw * H / W))
    head = 44

    def label(panel, text, sub):
        bar = np.full((head, pw, 3), 20, np.uint8)
        cv2.putText(bar, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, sub, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
        return np.concatenate([bar, panel], 0)

    def sub_of(m):
        s = summ[m]
        return f"PSNR {s['PSNR']:.2f}  SSIM {s['SSIM']:.3f}  LPIPS {s['LPIPS']:.3f}"

    tmp = a.out + ".frames"
    os.makedirs(tmp, exist_ok=True)
    for ci, p in enumerate(epos):
        ri = r_idx[p]
        M = np.vstack([map_s_to_g(A, ri, c, 2.0), [0, 0, 1]])
        if lowlight:
            (dx, dy), _ = cv2.phaseCorrelate(edges(mean_at(ref_root, p)), edges(mean_at(a.stack, p)))
            M = M @ np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]])
        gt = np.clip(cv2.warpAffine(comp, M[:2], (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REFLECT), 0, 255).astype(np.uint8)
        inp = cv2.imread(os.path.join(a.clip, f"{ci:05d}.png"))
        panels = [label(cv2.resize(np.clip(inp * gain, 0, 255).astype(np.uint8), (pw, ph), interpolation=cv2.INTER_AREA),
                        "Input" + (f" (x{gain:.1f} for display)" if lowlight else ""), sub_of("input_native") if "input_native" in summ else "")]
        for m in models:
            o = cv2.imread(os.path.join(a.runs, m, "frames", f"{ci:05d}.png"))
            panels.append(label(cv2.resize(np.clip(o * gain, 0, 255).astype(np.uint8), (pw, ph), interpolation=cv2.INTER_AREA),
                                NAMES[m], sub_of(m)))
        panels.append(label(cv2.resize(gt, (pw, ph), interpolation=cv2.INTER_AREA),
                            "GT: all-in-focus" + (" (lit scene 3)" if lowlight else ""), "best-focus composite"))
        row = np.concatenate([np.pad(q, ((0, 0), (0, 4), (0, 0))) for q in panels], 1)
        top = np.full((30, row.shape[1], 3), 0, np.uint8)
        cv2.putText(top, f"{a.title}   lens position {p} ({ci + 1}/{len(epos)})   worst -> best", (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.concatenate([top, row], 0)
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        cv2.imwrite(os.path.join(tmp, f"{ci:05d}.png"), frame)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
