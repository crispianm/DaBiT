"""Score the captured piano frames (no resampling) with run_clip.score; control: bicubic x2 must reproduce results.json."""
import sys, json, numpy as np, cv2
sys.path.insert(0, "/home/wg19671/repos/DaBiT/.claude/worktrees/clip-eval")
from run_clip import read_frames, score
import lpips as L
T = "/home/wg19671/repos/DaBiT/thesis_runs/piano_testset_20261112"
R = "/home/wg19671/repos/DaBiT/results_clip/piano_testset"
OUT = sys.argv[1]
dev = "cuda"
lp = L.LPIPS(net="alex", verbose=False).to(dev).eval()
res = {}
s = lambda d: {k: float(np.nanmean(v)) for k, v in d.items()}
for clip in ("1_24mm_bright", "2_24mm_low"):
    inp, _ = read_frames(f"{T}/{clip}/input")
    gt, _ = read_frames(f"{T}/{clip}/gt")
    n = min(len(inp), len(gt))
    inp, gt = inp[:n], gt[:n]
    w, h = 432, 288
    gt_eval = [cv2.resize(g, (2 * w, 2 * h)) for g in gt]
    up = [cv2.resize(cv2.resize(f, (w, h)), (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC) for f in inp]
    nat = score(inp, gt_eval, lp, dev)
    bic = score(up, gt_eval, lp, dev)
    ref = json.load(open(f"{R}/{clip}/results.json"))["summary"]["Input (bicubic x2)"]
    res[clip] = {"n": n, "input_native": s(nat), "input_bicubic_control": s(bic), "results_json_bicubic": ref,
                 "native_per_frame": {k: nat[k] for k in ("PSNR", "SSIM", "LPIPS")}}
    print(clip, n, "native", {k: round(v, 4) for k, v in s(nat).items()})
    print(clip, "bicubic control", {k: round(v, 4) for k, v in s(bic).items()}, "ref", {k: round(v, 4) for k, v in ref.items()})
json.dump(res, open(f"{OUT}/piano_native_scores.json", "w"), indent=1)
