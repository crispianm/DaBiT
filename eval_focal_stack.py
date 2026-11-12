# -*- coding: utf-8 -*-
"""Patch-level scoring of DaBiT on real focal stacks (bath_test).

A stack sweeps the lens over N positions with the camera fixed, so there is no
single sharp frame. Following the stack README, the reference for a patch is
the same patch at its best-focus slice (gt_pos, from gt/<stack>_gt.npz), averaged
over that slice's frames (almost noise free). Inputs are the clip frames from
make_stack_clips.py (one frame per position); outputs are run_clip.py frames,
resized from 2x the inference size back to the native frame size. Two
baselines: "input" goes through the model's resolution path (area-down to the
inference size, bicubic up), as in run_clip.py; "input_native" is the captured
frame itself, i.e. doing nothing.

Focus breathing: slices are not pixel aligned, so consecutive slice means are
registered with ECC (affine, half-resolution grey) and the warps are chained;
the reference is warped into the geometry of the scored slice. A control
reports raw-frame vs reference PSNR for patches within 1 slice of focus, with
and without the registration.

Scored pairs: valid patches (prominence > 3, gt_spread < 1.5) at lens positions
>= --min_dist slices (1 slice = 1365 lens units) from the patch's gt_pos.

Low light (--ref_stack): references, registration and GT come from a registered
lit stack (scene 4 -> scene 3). Each evaluated patch is gain-matched per channel
to the reference mean before scoring, so the score reflects structure rather
than the exposure difference.
"""
import os
import csv
import json
import argparse
from multiprocessing import get_context

import cv2
import numpy as np

SLICE = 1365.0
BINS = [(3, 6), (6, 12), (12, 1e9)]
METRICS = ("PSNR", "SSIM", "LPIPS")


def _ssim_psnr(chunk):
    from core.metrics import calc_psnr_and_ssim
    return [calc_psnr_and_ssim(a, b) for a, b in chunk]


def ecc_chain(small):
    """A[i]: 2x3 affine, coords of slice i -> coords of slice i+1 (capture order)."""
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    A, fails = [], 0
    for i in range(len(small) - 1):
        W = np.eye(2, 3, dtype=np.float32)
        try:
            _, W = cv2.findTransformECC(small[i], small[i + 1], W, cv2.MOTION_AFFINE, crit, None, 5)
        except cv2.error:
            W, fails = np.eye(2, 3, dtype=np.float32), fails + 1
        A.append(np.vstack([W.astype(np.float64), [0, 0, 1]]))
    return A, fails


def map_s_to_g(A, s, g, scale):
    """2x3 affine at full res: coords in slice s -> coords in slice g."""
    M = np.eye(3)
    if g > s:
        for i in range(s, g):
            M = A[i] @ M
    else:
        for i in range(s - 1, g - 1, -1):
            M = np.linalg.inv(A[i]) @ M
    S = np.diag([scale, scale, 1.0])
    return (S @ M @ np.linalg.inv(S))[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, help="Evaluated stack folder (meta.json + sPPPPP_fKK.png).")
    ap.add_argument("--gt", required=True, help="gt/<stack>_gt.npz")
    ap.add_argument("--clip", required=True, help="Clip frames from make_stack_clips.py")
    ap.add_argument("--runs", required=True, help="run_clip.py output folder")
    ap.add_argument("--models", nargs="+", default=["perceptual", "retrained", "paper"])
    ap.add_argument("--infer", default="512x336", help="Inference size used by run_clip.py")
    ap.add_argument("--ref_stack", default=None, help="Registered lit stack for low-light references")
    ap.add_argument("--min_dist", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    pool = get_context("fork").Pool(a.workers)  # fork before torch is imported
    import torch
    import lpips as lpips_lib
    from core.metrics import calculate_psnr

    ref_root = a.ref_stack or a.stack
    rmeta = json.load(open(os.path.join(ref_root, "meta.json")))
    emeta = json.load(open(os.path.join(a.stack, "meta.json")))
    F = rmeta["frames_per_slice"]
    rpos = [s["position"] for s in rmeta["slices"]]          # ref stack, capture order
    epos = [s["position"] for s in emeta["slices"]]          # evaluated stack, capture order
    r_idx = {p: i for i, p in enumerate(rpos)}
    e_idx = {p: i for i, p in enumerate(epos)}
    z = np.load(a.gt)
    positions = z["positions"].astype(np.int64)
    gt_pos, boxes = z["gt_pos"], z["boxes"]
    valid = np.where((z["prominence"] > 3) & (z["gt_spread"] < 1.5))[0]
    assert set(positions.tolist()) == set(rpos) == set(epos)

    def slice_mean(ri, root=None, p=None):
        root = root or ref_root
        p = rpos[ri] if p is None else p
        acc = sum(cv2.imread(os.path.join(root, f"s{p:05d}_f{f:02d}.png")).astype(np.float32) for f in range(F))
        return acc / F  # BGR

    def lighting_robust_edges(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(cv2.GaussianBlur(np.log1p(g), (0, 0), 2), cv2.CV_32F)

    def cross_stack_shift(p):
        """(dx, dy) of the evaluated stack relative to the ref stack at lens position p.
        The stack README says scenes 3 and 4 are pixel registered, but phase
        correlation finds a 4-6 px horizontal shift, so it is measured per position."""
        if a.ref_stack is None:
            return 0.0, 0.0
        (dx, dy), _ = cv2.phaseCorrelate(lighting_robust_edges(slice_mean(None, ref_root, p)),
                                         lighting_robust_edges(slice_mean(None, a.stack, p)))
        return dx, dy

    print("registering", ref_root, flush=True)
    small = []
    for ri in range(len(rpos)):
        g = cv2.cvtColor(slice_mean(ri), cv2.COLOR_BGR2GRAY)
        small.append(cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA))
    A, fails = ecc_chain(small)
    H, W = cv2.imread(os.path.join(a.clip, "00000.png")).shape[:2]
    iw, ih = (int(x) for x in a.infer.split("x"))

    def load_eval(m, ci):
        if m == "input_native":  # the clip frame as captured: the do-nothing baseline
            return cv2.imread(os.path.join(a.clip, f"{ci:05d}.png")).astype(np.float32)
        if m == "input":
            im = cv2.imread(os.path.join(a.clip, f"{ci:05d}.png"))
            im = cv2.resize(im, (iw, ih), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.imread(os.path.join(a.runs, m, "frames", f"{ci:05d}.png"))
        return cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    methods = ["input_native", "input"] + a.models
    lowlight = a.ref_stack is not None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = lpips_lib.LPIPS(net="alex", verbose=False).to(dev).eval()

    # best-focus slice (ref-stack index) of each valid patch
    g_of = {pi: r_idx[int(positions[np.argmin(np.abs(positions - gt_pos[pi]))])] for pi in valid}
    ref_cache = {}
    rows, async_res = [], []
    vals = {m: {"LPIPS": []} for m in methods}
    ctrl = {"registered": [], "unregistered": []}
    shifts = []

    for n, p in enumerate(positions):
        ci, ri = e_idx[int(p)], r_idx[int(p)]
        ev = {m: load_eval(m, ci) for m in methods}
        raw = cv2.imread(os.path.join(a.clip, f"{ci:05d}.png")).astype(np.float64)
        warped = {}
        batch = {m: [] for m in methods}
        dx, dy = cross_stack_shift(int(p))
        shifts.append((int(p), dx, dy))
        T = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)  # eval coords -> ref coords
        for pi in valid:
            d = abs(p - gt_pos[pi]) / SLICE
            if d >= a.min_dist or (d < 1.0 and not lowlight):
                gi = g_of[pi]
                if gi not in ref_cache:
                    ref_cache[gi] = slice_mean(gi)
                if gi not in warped:
                    M = np.vstack([map_s_to_g(A, ri, gi, 2.0), [0, 0, 1]]) @ T
                    warped[gi] = np.clip(cv2.warpAffine(
                        ref_cache[gi], M[:2], (W, H),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT), 0, 255)
                x0, y0, x1, y1 = boxes[pi]
                rp = warped[gi][y0:y1, x0:x1]
            if d < 1.0 and not lowlight:  # registration control
                ctrl["registered"].append(calculate_psnr(raw[y0:y1, x0:x1], rp.astype(np.float64)))
                ctrl["unregistered"].append(calculate_psnr(
                    raw[y0:y1, x0:x1], np.clip(ref_cache[gi][y0:y1, x0:x1], 0, 255).astype(np.float64)))
            if d < a.min_dist:
                continue
            rows.append({"position": int(p), "clip_frame": ci, "patch": int(pi), "dist": float(d)})
            for m in methods:
                ep = ev[m][y0:y1, x0:x1]
                if lowlight:
                    gain = rp.reshape(-1, 3).mean(0) / np.maximum(ep.reshape(-1, 3).mean(0), 1e-3)
                    ep = np.clip(ep * gain, 0, 255)
                batch[m].append((np.ascontiguousarray(ep[..., ::-1]), np.ascontiguousarray(rp[..., ::-1])))
        if batch["input"]:
            for m in methods:
                async_res.append((m, pool.map_async(_ssim_psnr, [batch[m]])))
                with torch.no_grad():
                    c = torch.from_numpy(np.stack([q[0] for q in batch[m]]) / 255.0).permute(0, 3, 1, 2).float().to(dev)
                    g = torch.from_numpy(np.stack([q[1] for q in batch[m]]) / 255.0).permute(0, 3, 1, 2).float().to(dev)
                    vals[m]["LPIPS"].extend(lp(c, g, normalize=True).view(-1).cpu().tolist())
        if n % 16 == 0:
            print(f"  position {n}/{len(positions)}: {len(rows)} pairs", flush=True)

    for m in methods:
        vals[m]["PSNR"], vals[m]["SSIM"] = [], []
    for m, r in async_res:
        for ps, ss in r.get()[0]:
            vals[m]["PSNR"].append(ps)
            vals[m]["SSIM"].append(ss)
    pool.close()
    for m in methods:
        for k in METRICS:
            vals[m][k] = np.array(vals[m][k])
            assert len(vals[m][k]) == len(rows)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "patch_metrics.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["position", "clip_frame", "patch", "dist_slices"] + [f"{m}_{k}" for m in methods for k in METRICS])
        for j, r in enumerate(rows):
            wr.writerow([r["position"], r["clip_frame"], r["patch"], f"{r['dist']:.3f}"] +
                        [f"{vals[m][k][j]:.5f}" for m in methods for k in METRICS])
    dist = np.array([r["dist"] for r in rows])
    summ = {"n_pairs": len(rows), "n_valid_patches": int(len(valid)), "ecc_failures": fails,
            "lowlight_gain_matched": lowlight, "min_dist_slices": a.min_dist,
            "cross_stack_shift_px": ({"dx_median": float(np.median([s[1] for s in shifts])),
                                      "dx_range": [float(min(s[1] for s in shifts)), float(max(s[1] for s in shifts))],
                                      "dy_median": float(np.median([s[2] for s in shifts]))} if lowlight else None),
            "control_psnr_within_1_slice": {k: (float(np.mean(v)) if v else None) for k, v in ctrl.items()},
            "all": {m: {k: float(vals[m][k].mean()) for k in METRICS} for m in methods}, "bins": {}}
    for lo, hi in BINS:
        sel = (dist >= lo) & (dist < hi)
        key = f"{lo}-{int(hi) if hi < 1e8 else 'inf'}"
        summ["bins"][key] = {"n": int(sel.sum())}
        for m in methods:
            summ["bins"][key][m] = {k: (float(vals[m][k][sel].mean()) if sel.any() else None) for k in METRICS}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps({k: summ[k] for k in ("n_pairs", "ecc_failures", "control_psnr_within_1_slice", "all")}, indent=1))


if __name__ == "__main__":
    main()
