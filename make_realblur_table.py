"""Rows of the merged real-blur table (thesis Ch 3, dabit:tbl:realblur) from the source JSONs; prints LaTeX rows + gains."""
import json
B = json.load(open("/home/wg19671/repos/DaBiT/results_clip/bath_test/summary_thesis_filter.json"))["per_scene"]
PN = json.load(open("/home/wg19671/repos/DaBiT/results_clip/piano_testset/piano_native_scores.json"))
PR = "/home/wg19671/repos/DaBiT/results_clip/piano_testset"
M = ["input_native", "paper", "retrained", "perceptual"]


def piano(clip):
    s = json.load(open(f"{PR}/{clip}/results.json"))["summary"]
    d = {m: s[m] for m in ("paper", "retrained", "perceptual")}
    d["input_native"] = PN[clip]["input_native"]
    return d


rows = [("Normal light", None, None),
        ("Piano", "p", "1_24mm_bright"), ("Living room", "b", "Scene 1 (living room)"),
        ("Near content", "b", "Scene 2 (near content)"), ("Hallway", "b", "Scene 3 (hallway, lit)"),
        ("Kitchen", "b", "Scene 5 (kitchen, lit)"),
        ("Low light", None, None),
        ("Piano", "p", "2_24mm_low"), ("Hallway", "b", "Scene 4 (hallway, lights off, ISO 12800)*"),
        ("Kitchen", "b", "Scene 5 (kitchen, lights off, ISO 12800)*")]
for name, kind, key in rows:
    if kind is None:
        print(r"\midrule" + "\n" + rf"\multicolumn{{9}}{{@{{}}l}}{{\textit{{{name}}}}} \\")
        continue
    d = piano(key) if kind == "p" else B[key]
    P = {m: round(d[m]["PSNR"], 2) for m in M}
    L = {m: round(d[m]["LPIPS"], 3) for m in M}
    bp, bl = max(P.values()), min(L.values())
    cells = []
    for m in M:
        cells.append((r"\textbf{%.2f}" if P[m] == bp else "%.2f") % P[m])
        cells.append((r"\textbf{%.3f}" if L[m] == bl else "%.3f") % L[m])
    print(f"{name} & " + " & ".join(cells) + r" \\")
    gi = d["input_native"]
    print("%   gains vs input: " + ", ".join(f"{m} {d[m]['PSNR'] - gi['PSNR']:+.3f} dB LPIPS {100 * (1 - d[m]['LPIPS'] / gi['LPIPS']):.1f}%"
                                         for m in M[1:]))
