# -*- coding: utf-8 -*-
"""Thesis Ch 3 qualitative figure for all real-blur captures (piano focus-pull + bath_test focal stacks).

Rows: one capture each, lit (piano 24 mm, scenes 1, 2, 3, 5) then low light (scene 4, scene 5 ISO 12800).
Columns: captured input, DaBiT (paper), DaBiT (retrained), DaBiT (perceptual), reference.
Style follows the thesis Ch 2 redraws: Latin Modern Roman, bold labels, rounded black-bordered tiles.

Frame choice (best case, as make_figure.py): the frame where the best model gains most PSNR over the captured input.
  Focal stacks: means over that frame's patches >= 12 slices from best focus, thesis patch filter, frames with >= 20 such
  patches.  Piano: full-frame per-frame scores (captured input from piano_native_scores.json, models from per_frame.csv).
Per-tile metrics are those means (PSNR dB / LPIPS AlexNet); best per row in bold.

Run: OMP_NUM_THREADS=1 python make_realblur_figure.py PIANO_NATIVE_JSON OUTDIR [THESIS_FIG_DIR]
"""
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_focal_stack import ecc_chain, map_s_to_g, patch_filter  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

R = "/home/wg19671/repos/DaBiT/results_clip/bath_test"
PR = "/home/wg19671/repos/DaBiT/results_clip/piano_testset"
PT = "/home/wg19671/repos/DaBiT/thesis_runs/piano_testset_20261112"
STK = "/home/wg19671/repos/dabit/bath_test"
EV = "ev" + "al"
PIANO_JSON, OUT = sys.argv[1], sys.argv[2]
THESIS = sys.argv[3] if len(sys.argv) > 3 else None
CACHE = os.path.join(R, "figure", "_cache")  # composites shared with make_figure.py
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

# (row label, kind, stack or clip, lit reference stack for low light, group)
ROWS = [("Piano", "piano", "1_24mm_bright", None, "Normal light"),
        ("Living room", "stack", "stack_forward1", None, "Normal light"),
        ("Near content", "stack", "scene2_forward", None, "Normal light"),
        ("Hallway", "stack", "scene3_forward", None, "Normal light"),
        ("Kitchen", "stack", "scene5_lit_forward", None, "Normal light"),
        ("Hallway", "stack", "scene4_forward", "scene3_forward", "Low light"),
        ("Kitchen", "stack", "scene5_dark_forward", "scene5_lit_forward", "Low light")]
MODELS = ["paper", "retrained", "perceptual"]
KEYS = ["input_native"] + MODELS
COLHEAD = ["Captured input", "Paper", "Retrained", "Perceptual", "Reference"]  # cols 2-4 under a "DaBiT" spanner
COLS = ["position", "clip_frame", "patch", "dist_slices"] + \
       [f"{m}_{k}" for m in ("input_native", "input", "perceptual", "retrained", "paper") for k in ("PSNR", "SSIM", "LPIPS")]
MIN_DIST, MIN_PATCHES = 12.0, 20
ZW, ZH = 96, 64            # zoom window (3:2) in source pixels
ACCENT = "#C2185B"         # Ch 2 redraw magenta: zoom box and inset border
INK, MUTED = "#1a1a1a", "#555555"

for w in ("regular", "bold"):
    fm.fontManager.addfont(os.path.expanduser(f"~/.local/share/fonts/lmroman10-{w}.otf"))
plt.rcParams.update({"font.family": "Latin Modern Roman", "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": INK})


def stack_dir(name):
    return f"{STK}/scene5_kitchen/{name}" if name.startswith("scene5") else f"{STK}/{name}"


def read_csv(stack, gt_stack):
    with open(f"{R}/{EV}/{stack}/patch_metrics.csv") as fh:
        rd = csv.reader(fh)
        assert next(rd) == COLS
        d = np.array([[float(x) for x in r] for r in rd])
    keep = patch_filter(np.load(f"{STK}/gt/{gt_stack}_gt.npz"), "thesis")
    return d[np.isin(d[:, 2].astype(int), keep)]


def col(d, m, k):
    return d[:, COLS.index(f"{m}_{k}")]


def composite(ref_stack):
    """All-in-focus composite + chained ECC; same code and cache as make_figure.py."""
    cp = f"{CACHE}/comp_{ref_stack}.npz"
    if os.path.exists(cp):
        z = np.load(cp)
        return z["comp"], z["A"]
    root = stack_dir(ref_stack)
    meta = json.load(open(f"{root}/meta.json"))
    F = meta["frames_per_slice"]
    rpos = [s["position"] for s in meta["slices"]]
    means = [sum(cv2.imread(f"{root}/s{p:05d}_f{f:02d}.png").astype(np.float32) for f in range(F)) / F for p in rpos]
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
    A = np.array(A)
    np.savez_compressed(cp, comp=comp, A=A)
    return comp, A


def edges(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(cv2.GaussianBlur(np.log1p(g), (0, 0), 2), cv2.CV_32F)


def zoom_window(imgs, gt, bm, centres, lowlight, pos_limit):
    """Window with the largest best-model-over-input L1 gain among textured, unsaturated windows."""
    H, W = gt.shape[:2]
    tex, sat, wing, mid, cc, xy, gmid = [], [], [], [], [], [], []
    for cx, cy in centres:
        wx = int(np.clip(cx - ZW // 2, 0, W - ZW))
        wy = int(np.clip(cy - ZH // 2, 0, H - ZH))
        sl = (slice(wy, wy + ZH), slice(wx, wx + ZW))
        g = cv2.cvtColor(gt[sl], cv2.COLOR_BGR2GRAY)
        tex.append(cv2.Laplacian(g, cv2.CV_32F).var())
        wing.append(np.abs(imgs["input_native"][sl].astype(np.float32) - gt[sl]).mean()
                    - np.abs(imgs[bm][sl].astype(np.float32) - gt[sl]).mean())
        sat.append(max((imgs[k][sl] >= 250).mean() for k in imgs))
        ig = cv2.cvtColor(imgs["input_native"][sl], cv2.COLOR_BGR2GRAY).ravel().astype(float)
        cc.append(np.corrcoef(ig, g.ravel().astype(float))[0, 1])
        mid.append(imgs["input_native"][sl].mean())
        gmid.append(gt[sl].mean())
        xy.append((wx, wy))
    tex, sat, wing, mid, cc, gmid = map(np.array, (tex, sat, wing, mid, cc, gmid))
    cc = np.nan_to_num(cc)
    ok = (tex >= np.percentile(tex, 75)) & (sat < 0.01)
    if lowlight:  # dark mid-tone content with the same structure in lit reference and dark input
        base = (sat < 0.01) & (mid > 50) & (mid < 180) & (gmid > 50) & (gmid < 200) & (cc > 0.3)
        if pos_limit:  # scene 4: keep to the dark near side (as make_figure.py)
            base &= np.array([(x + ZW / 2) < 0.5 * W and (y + ZH / 2) < 0.55 * H for x, y in xy])
        if base.sum() == 0:
            base = sat < 0.01
        ok = base & (tex >= np.percentile(tex[base], 50))
    j = int(np.argmax(np.where(ok, wing, -1e9)))
    return xy[j], float(wing[j])


def stack_panel(stack, ref):
    gt_stack = ref or stack
    d = read_csv(stack, gt_stack)
    cand = []
    for f in np.unique(d[:, 1]):
        s = d[(d[:, 1] == f) & (d[:, 3] >= MIN_DIST)]
        if len(s) < MIN_PATCHES:
            continue
        inp = col(s, "input_native", "PSNR").mean()
        pm = {m: col(s, m, "PSNR").mean() for m in MODELS}
        bm = max(pm, key=pm.get)
        cand.append((pm[bm] - inp, int(f), bm, len(s)))
    cand.sort(reverse=True)
    gain_db, f, bm, n = cand[0]
    s = d[(d[:, 1] == f) & (d[:, 3] >= MIN_DIST)]
    meta = json.load(open(f"{stack_dir(stack)}/meta.json"))
    rmeta = json.load(open(f"{stack_dir(gt_stack)}/meta.json"))
    p = [q["position"] for q in meta["slices"]][f]
    rpos = [q["position"] for q in rmeta["slices"]]
    ri = rpos.index(p)
    comp, A = composite(gt_stack)
    A = [a for a in A]
    H, W = comp.shape[:2]
    M = np.vstack([map_s_to_g(A, ri, len(rpos) // 2, 2.0), [0, 0, 1]])
    info = dict(clip_frame=f, best_model=bm, psnr_gain_best_model=gain_db, n_frames_eligible=len(cand),
                lens_position=int(p), n_patches_ge12=int(n))
    if ref:
        Fr, Fe = rmeta["frames_per_slice"], meta["frames_per_slice"]
        mean_at = lambda root, F: sum(cv2.imread(f"{root}/s{p:05d}_f{k:02d}.png").astype(np.float32) for k in range(F)) / F  # noqa: E731
        (dx, dy), _ = cv2.phaseCorrelate(edges(mean_at(stack_dir(gt_stack), Fr)), edges(mean_at(stack_dir(stack), Fe)))
        M = M @ np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]])
        info["shift_px"] = [float(dx), float(dy)]
    gt = np.clip(cv2.warpAffine(comp, M[:2], (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REFLECT), 0, 255).astype(np.uint8)
    imgs = {"input_native": cv2.imread(f"{R}/clips/{stack}/{f:05d}.png")}
    for m in MODELS:
        imgs[m] = cv2.resize(cv2.imread(f"{R}/runs/{stack}/{m}/frames/{f:05d}.png"), (W, H), interpolation=cv2.INTER_CUBIC)
    disp = 1.0
    if ref:  # one shared display gain for input and model panels (make_stack_video.py rule)
        x = cv2.imread(f"{R}/clips/{stack}/00064.png").astype(np.float32).mean()
        disp = round(float(comp.mean() / max(x, 1e-3)), 1)
        for k in KEYS:
            imgs[k] = np.clip(imgs[k].astype(np.float32) * disp, 0, 255).astype(np.uint8)
    imgs["gt"] = gt
    z = np.load(f"{STK}/gt/{gt_stack}_gt.npz")
    boxes = z["boxes"]
    pids = patch_filter(z, "thesis") if ref else s[:, 2].astype(int)
    centres = [((boxes[q][0] + boxes[q][2]) // 2, (boxes[q][1] + boxes[q][3]) // 2) for q in pids]
    (rx, ry), wl1 = zoom_window(imgs, gt, bm, centres, bool(ref), stack == "scene4_forward")
    mets = {m: {k: float(col(s, m, k).mean()) for k in ("PSNR", "SSIM", "LPIPS")} for m in KEYS}
    info.update(zoom_box_xywh=[rx, ry, ZW, ZH], zoom_window_l1_gain=wl1, display_gain=(disp if ref else None),
                metrics_over="patches >= 12 slices from best focus (thesis filter)", metrics=mets)
    return dict(imgs=imgs, mets=mets, box=(rx, ry), disp=disp, frame=f, W=W, H=H), info


def piano_panel(clip):
    nat = json.load(open(PIANO_JSON))[clip]["native_per_frame"]
    pf = {}
    with open(f"{PR}/{clip}/per_frame.csv") as fh:
        for r in csv.DictReader(fh):
            pf[(r["method"], int(r["frame"]))] = r
    n = len(nat["PSNR"])
    cand = []
    for f in range(n):
        pm = {m: float(pf[(m, f)]["PSNR"]) for m in MODELS}
        bm = max(pm, key=pm.get)
        cand.append((pm[bm] - nat["PSNR"][f], f, bm))
    cand.sort(reverse=True)
    gain_db, f, bm = cand[0]
    imgs = {"input_native": cv2.imread(sorted(glob.glob(f"{PT}/{clip}/input/*"))[f]),
            "gt": cv2.imread(sorted(glob.glob(f"{PT}/{clip}/gt/*"))[f])}
    H, W = imgs["gt"].shape[:2]
    for m in MODELS:
        imgs[m] = cv2.resize(cv2.imread(f"{PR}/{clip}/{m}/frames/{f:05d}.png"), (W, H), interpolation=cv2.INTER_CUBIC)
    centres = [(x, y) for y in range(ZH // 2, H - ZH // 2, 16) for x in range(ZW // 2, W - ZW // 2, 16)]
    (rx, ry), wl1 = zoom_window(imgs, imgs["gt"], bm, centres, False, False)
    mets = {"input_native": {k: float(nat[k][f]) for k in ("PSNR", "SSIM", "LPIPS")}}
    for m in MODELS:
        mets[m] = {k: float(pf[(m, f)][k]) for k in ("PSNR", "SSIM", "LPIPS")}
    info = dict(clip_frame=f, best_model=bm, psnr_gain_best_model=gain_db, n_frames_eligible=n,
                zoom_box_xywh=[rx, ry, ZW, ZH], zoom_window_l1_gain=wl1, metrics_over="full frame", metrics=mets)
    return dict(imgs=imgs, mets=mets, box=(rx, ry), disp=1.0, frame=f, W=W, H=H), info


def rounded_image(fig, rect, img, W, H, box=None, radius=0.07):
    """Image in a rounded black-bordered tile (Ch 2 style); optional zoom box + inset."""
    a = fig.add_axes(rect)
    a.set_axis_off()
    a.set_xlim(0, W)
    a.set_ylim(H, 0)
    clip = FancyBboxPatch((0, 0), W, H, boxstyle=f"round,pad=0,rounding_size={radius * H}", transform=a.transData,
                          fc="none", ec="none")
    a.add_patch(clip)
    im = a.imshow(img, extent=(0, W, H, 0), aspect="auto", interpolation="lanczos")
    im.set_clip_path(clip)
    if box is not None:
        rx, ry = box
        a.add_patch(Rectangle((rx, ry), ZW, ZH, fill=False, ec=ACCENT, lw=0.9, zorder=3))
        iw = 0.42 * W
        ih = iw * ZH / ZW
        mg = 0.035 * W
        corners = [(W - iw - mg, H - ih - mg), (mg, H - ih - mg), (W - iw - mg, mg), (mg, mg)]
        pos = corners[0]
        for cx0, cy0 in corners:
            if rx + ZW + 10 < cx0 or rx > cx0 + iw + 10 or ry + ZH + 10 < cy0 or ry > cy0 + ih + 10:
                pos = (cx0, cy0)
                break
        crop = cv2.resize(img[ry:ry + ZH, rx:rx + ZW], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        a.imshow(crop, extent=(pos[0], pos[0] + iw, pos[1] + ih, pos[1]), aspect="auto", interpolation="lanczos", zorder=4)
        a.add_patch(Rectangle(pos, iw, ih, fill=False, ec=ACCENT, lw=1.1, zorder=5))
    a.add_patch(FancyBboxPatch((0, 0), W, H, boxstyle=f"round,pad=0,rounding_size={radius * H}", fc="none", ec="black",
                               lw=1.3, zorder=6, clip_on=False))
    return a


def draw(panels):
    cm = 1 / 2.54
    FW = 15.5 * cm
    grp_w, lab_w, gut = 0.42 * cm, 0.36 * cm, 0.12 * cm
    nc = len(COLHEAD)
    tw = (FW - grp_w - lab_w - (nc - 1) * gut) / nc
    th = tw / 1.5
    head_h, met_h, rgap, grp_gap = 0.92 * cm, 0.36 * cm, 0.10 * cm, 0.22 * cm
    FH = head_h + len(ROWS) * (th + met_h + rgap) + grp_gap
    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")
    X0 = grp_w + lab_w
    for ci, h in enumerate(COLHEAD):
        fig.text((X0 + ci * (tw + gut) + tw / 2) / FW, 1 - (0.12 * cm if ci in (0, 4) else 0.50 * cm) / FH, h,
                 ha="center", va="top" if ci in (0, 4) else "top", fontsize=8.5, fontweight="bold")
    xs, xe = X0 + 1 * (tw + gut), X0 + 3 * (tw + gut) + tw  # DaBiT spanner over the three model columns
    fig.text((xs + xe) / 2 / FW, 1 - 0.08 * cm / FH, "DaBiT", ha="center", va="top", fontsize=8.5, fontweight="bold")
    fig.add_artist(plt.Line2D([(xs + 0.1 * cm) / FW, (xe - 0.1 * cm) / FW], [1 - 0.43 * cm / FH] * 2, color="black", lw=1.3))
    y = head_h
    gspan = {}
    for ri, (label, kind, name, ref, grp) in enumerate(ROWS):
        if ri > 0 and grp != ROWS[ri - 1][4]:
            y += grp_gap
        P = panels[name]
        for ci, key in enumerate(KEYS + ["gt"]):
            x = X0 + ci * (tw + gut)
            img = cv2.cvtColor(P["imgs"][key], cv2.COLOR_BGR2RGB)
            rounded_image(fig, [x / FW, 1 - (y + th) / FH, tw / FW, th / FH], img, P["W"], P["H"], P["box"])
            if key != "gt":
                m = P["mets"][key]
                bp = max(P["mets"][q]["PSNR"] for q in KEYS)
                bl = min(P["mets"][q]["LPIPS"] for q in KEYS)
                cy = 1 - (y + th + 0.07 * cm) / FH
                fig.text((x + tw * 0.47) / FW, cy, f"{m['PSNR']:.2f} dB", ha="right", va="top", fontsize=7,
                         fontweight="bold" if abs(m["PSNR"] - bp) < 1e-12 else "normal")
                fig.text((x + tw * 0.53) / FW, cy, f"{m['LPIPS']:.3f}", ha="left", va="top", fontsize=7,
                         fontweight="bold" if abs(m["LPIPS"] - bl) < 1e-12 else "normal")
                fig.text((x + tw * 0.50) / FW, cy, "/", ha="center", va="top", fontsize=7, color=MUTED)
        fig.text((grp_w + lab_w * 0.45) / FW, 1 - (y + th / 2) / FH, label, rotation=90, ha="center", va="center",
                 fontsize=8, fontweight="bold")
        gspan.setdefault(grp, [y, y + th])[1] = y + th
        y += th + met_h + rgap
    for grp, (y0, y1) in gspan.items():  # group bracket + label, like the table's Lit / Low light blocks
        xb = grp_w * 0.78
        fig.add_artist(plt.Line2D([xb / FW, xb / FW], [1 - y0 / FH, 1 - y1 / FH], color="black", lw=1.3))
        fig.text(grp_w * 0.32 / FW, 1 - (y0 + y1) / 2 / FH, grp, rotation=90, ha="center", va="center", fontsize=8.5,
                 fontweight="bold")
    outs = [OUT] + ([THESIS] if THESIS else [])
    for o in outs:
        fig.savefig(f"{o}/realblur_qualitative.pdf", facecolor="white")
    fig.savefig(f"{OUT}/realblur_qualitative.png", dpi=300, facecolor="white")
    print("figure size cm:", FW / cm, FH / cm)


def main():
    panels, sel = {}, {}
    for label, kind, name, ref, grp in ROWS:
        P, info = piano_panel(name) if kind == "piano" else stack_panel(name, ref)
        info.update(row=label, group=grp, reference=(ref or ("in-focus clip" if kind == "piano" else name)))
        panels[name], sel[name] = P, info
        print(name, "frame", P["frame"] + 1, info["best_model"], f"gain {info['psnr_gain_best_model']:.2f} dB",
              "box", info["zoom_box_xywh"], "disp", P["disp"], flush=True)
    draw(panels)
    json.dump(sel, open(f"{OUT}/realblur_figure_selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
