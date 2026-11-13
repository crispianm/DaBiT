# -*- coding: utf-8 -*-
"""Re-aggregate bath_test focal-stack scores under a patch filter.

Reads each eval/<stack>/patch_metrics.csv (from eval_focal_stack.py) and the
stack's GT npz, keeps the rows whose patch passes --filter (patch_filter in
eval_focal_stack.py), and writes per-scene, by-distance, per-stack and low-light
ISO tables. Aggregation as the 11-12 summarise.py: each stack's mean over its
patch pairs, then the unweighted mean over stacks (no pooling over pairs).

Scenes 1-4 CSVs were made under the old key filter (prominence > 3,
gt_spread < 1.5); the thesis filter is a strict subset, so they are filtered
here without rerunning. Scene-5 CSVs were made under the thesis filter, so they
have no key-filter values.

    python summarize_focal_stacks.py --root results_clip/bath_test            # writes SUMMARY.md + json
    python summarize_focal_stacks.py --root ... --filter key --print_only     # reproduction control
    python summarize_focal_stacks.py --root ... --gt_override scene3_forward=scene2_forward --print_only  # negative control
"""
import os
import csv
import json
import argparse

import numpy as np

from eval_focal_stack import BINS, METRICS, SLICE, patch_filter

GT = os.path.expanduser("~/repos/dabit/bath_test/gt")
M = ["input_native", "input", "perceptual", "retrained", "paper"]
LAB = {"input_native": "Input (as captured)", "input": "Input (512x336, bicubic x2)", "perceptual": "perceptual",
       "retrained": "retrained", "paper": "paper"}
SC = {"Scene 1 (living room)": ["stack_forward1", "stack_backward1", "stack_forward2"],
      "Scene 2 (near content)": ["scene2_forward", "scene2_backward"],
      "Scene 3 (hallway, lit)": ["scene3_forward", "scene3_backward", "scene3_forward2"],
      "Scene 4 (hallway, lights off, ISO 12800)*": ["scene4_forward"],
      "Scene 5 (kitchen, lit)": ["scene5_lit_forward", "scene5_lit_backward"],
      "Scene 5 (kitchen, lights off, ISO 12800)*": ["scene5_dark_forward"]}
LIT = SC["Scene 1 (living room)"] + SC["Scene 2 (near content)"] + SC["Scene 3 (hallway, lit)"] + SC["Scene 5 (kitchen, lit)"]
ISO = {12800: "scene5_dark_forward", 25600: "scene5_dark_iso25600", 51200: "scene5_dark_iso51200",
       102400: "scene5_dark_iso102400", 204800: "scene5_dark_iso204800", 409600: "scene5_dark_iso409600"}
GT_OF = {"scene4_forward": "scene3_forward", **{s: "scene5_lit_forward" for s in ISO.values()}}
BKEYS = [f"{lo}-{int(hi) if hi < 1e8 else 'inf'}" for lo, hi in BINS]


def stack_stats(root, s, filt, gt_override):
    p = os.path.join(root, "eval", s, "patch_metrics.csv")
    if not os.path.isfile(p):
        return None
    g = gt_override.get(s, GT_OF.get(s, s))
    z = np.load(os.path.join(GT, f"{g}_gt.npz"))
    keep = set(patch_filter(z, filt).tolist())
    rows = [r for r in csv.DictReader(open(p)) if int(r["patch"]) in keep]
    # exact distance (the CSV rounds to 3 dp, which flips a few pairs at the bin edges)
    dist = np.array([abs(int(r["position"]) - z["gt_pos"][int(r["patch"])]) / SLICE for r in rows])
    v = {m: {k: np.array([float(r[f"{m}_{k}"]) for r in rows]) for k in METRICS} for m in M}
    out = {"n_pairs": len(rows), "n_patches": len({r["patch"] for r in rows}),
           "all": {m: {k: float(v[m][k].mean()) if rows else None for k in METRICS} for m in M}, "bins": {}}
    summ = json.load(open(os.path.join(root, "eval", s, "summary.json")))
    out["ecc_failures"] = summ["ecc_failures"]
    out["cross_stack_shift_px"] = summ.get("cross_stack_shift_px")
    out["control_psnr_within_1_slice"] = summ.get("control_psnr_within_1_slice")
    for (lo, hi), b in zip(BINS, BKEYS):
        sel = (dist >= lo) & (dist < hi)
        out["bins"][b] = {"n": int(sel.sum()),
                          **{m: {k: float(v[m][k][sel].mean()) if sel.any() else None for k in METRICS} for m in M}}
    return out


def mean_over(S, stacks, part, m):
    vs = [(S[s]["all"] if part == "all" else S[s]["bins"][part])[m] for s in stacks]
    return {k: sum(x[k] for x in vs) / len(vs) for k in METRICS}


def table(vals, title, cols=METRICS):
    fmt = {"PSNR": "{:.2f}", "SSIM": "{:.4f}", "LPIPS": "{:.4f}"}
    arrow = {"PSNR": "↑", "SSIM": "↑", "LPIPS": "↓"}
    out = [f"### {title}", "| Method | " + " | ".join(f"{k} {arrow[k]}" for k in cols) + " |",
           "|---|" + "---:|" * len(cols)]
    best = {k: fmt[k].format((max if k != "LPIPS" else min)(vals[m][k] for m in M)) for k in cols}
    for m in M:
        cells = [fmt[k].format(vals[m][k]) for k in cols]
        out.append(f"| {LAB[m]} | " + " | ".join(f"**{c}**" if c == best[k] else c for c, k in zip(cells, cols)) + " |")
    return "\n".join(out) + "\n"


def build(root, filt, gt_override):
    S = {s: stack_stats(root, s, filt, gt_override) for st in SC.values() for s in st}
    S.update({s: stack_stats(root, s, filt, gt_override) for s in ISO.values()})
    S = {s: x for s, x in S.items() if x is not None and x["n_pairs"]}
    res = {"filter": filt, "aggregation": "unweighted mean over stacks of per-stack means over patch pairs",
           "per_stack": S, "per_scene": {}, "lit_by_distance": {}, "lowlight_iso": {}}
    lines = []
    lines.append("## Mean per scene (all scored patches)\n")
    for sc, st in SC.items():
        st = [s for s in st if s in S]
        if not st:
            continue
        res["per_scene"][sc] = {m: mean_over(S, st, "all", m) for m in M}
        lines.append(table(res["per_scene"][sc], f"{sc}: {', '.join(st)}"))
    lit = [s for s in LIT if s in S]
    res["lit_stacks"] = lit
    res["lit_all"] = {m: mean_over(S, lit, "all", m) for m in M}
    lines.append(f"## All lit stacks ({len(lit)}), mean over stacks\n")
    lines.append(table(res["lit_all"], f"all scored patches ({sum(S[s]['n_pairs'] for s in lit)} pairs)"))
    lines.append(f"## By distance from best focus (mean over the {len(lit)} lit stacks)\n")
    for b in BKEYS:
        n = sum(S[s]["bins"][b]["n"] for s in lit)
        res["lit_by_distance"][b] = {"n": n, **{m: mean_over(S, lit, b, m) for m in M}}
        lines.append(table(res["lit_by_distance"][b], f"{b} slices from focus ({n} patch pairs)"))
    lines.append("## Low light by ISO (scene 4: hallway; scene 5: kitchen; vs lit references)\n")
    rows_ll = [("scene 4", 12800, "scene4_forward")] + [("scene 5", i, s) for i, s in ISO.items()]
    for part in ("12-inf", "all"):
        hdr = "| Stack | ISO | pairs | " + " | ".join(f"{LAB[m]} PSNR / LPIPS" for m in M) + " |"
        lines += [f"### {part} slices from focus" if part != "all" else "### all scored patches", hdr,
                  "|---|---:|---:|" + "---|" * len(M)]
        for sc, iso, s in rows_ll:
            if s not in S:
                continue
            x = S[s]["all"] if part == "all" else S[s]["bins"][part]
            n = S[s]["n_pairs"] if part == "all" else x["n"]
            res["lowlight_iso"].setdefault(s, {"scene": sc, "iso": iso})[part] = {"n": n, **{m: x[m] for m in M}}
            lines.append(f"| {sc} | {iso} | {n} | " +
                         " | ".join(f"{x[m]['PSNR']:.2f} / {x[m]['LPIPS']:.4f}" for m in M) + " |")
        lines.append("")
    lines.append("## Per stack (all scored patches)\n")
    for s, x in S.items():
        lines.append(table(x["all"], f"{s} ({x['n_pairs']} pairs, {x['n_patches']} patches, ECC failures {x['ecc_failures']})"))
    return res, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/wg19671/repos/DaBiT/results_clip/bath_test")
    ap.add_argument("--filter", choices=["thesis", "key"], default="thesis")
    ap.add_argument("--gt_override", nargs="*", default=[], help="stack=gtname (negative control)")
    ap.add_argument("--print_only", action="store_true")
    a = ap.parse_args()
    ov = dict(x.split("=") for x in a.gt_override)
    res, lines = build(a.root, a.filter, ov)
    if a.print_only:
        print("\n".join(lines))
        json.dump(res, open(os.devnull, "w"))
        return
    old, old_lines = build(a.root, "key", ov)
    head = ["# DaBiT on bath_test real focal stacks (scenes 1-4 run 2026-11-12, scene 5 2026-11-13)", "",
            "Built by `summarize_focal_stacks.py` from each `eval/<stack>/patch_metrics.csv`. Thesis filter (Ch 3 and Ch 6): "
            "median over the 4 GT measures of max/median of `clean_<m>` > 3, `gt_spread` < 1.5, `gt_pos` < 63487. "
            "Means: per stack over patch pairs, then unweighted over stacks. Bold = best per column.", "",
            "\\* Low light is scored against the same scene's lit forward references (scene 4 vs scene 3, scene 5 dark vs "
            "scene 5 lit forward), gain-matched per patch and shift-corrected per lens position. Scene-5 lit forward "
            "references are 30-frame means (its frames_per_slice).", ""]
    tail = ["", "---", "", "# Old key filter (prominence > 3, gt_spread < 1.5), for the old-filter clause",
            "Scene 5 was scored under the thesis filter only, so its rows here equal the thesis-filter rows.", ""]
    open(os.path.join(a.root, "SUMMARY.md"), "w").write("\n".join(head + lines + tail + old_lines))
    json.dump(res, open(os.path.join(a.root, "summary_thesis_filter.json"), "w"), indent=1)
    json.dump(old, open(os.path.join(a.root, "summary_key_filter.json"), "w"), indent=1)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
