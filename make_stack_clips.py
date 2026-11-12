# -*- coding: utf-8 -*-
"""Turn bath_test focal stacks into focus-pull clips for run_clip.py.

Each stack has N lens positions x F frames per position. The clip keeps the
whole sweep (every lens position, first to last: nothing is trimmed) and
shortens it by taking one frame per position (--frame, default 5, i.e. the
middle of the 10 settled frames). Frames are symlinked as 00000.png, ...

    python make_stack_clips.py --stacks ~/repos/dabit/bath_test --out results_clip/bath_test/clips
"""
import os
import json
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=5)
    a = ap.parse_args()
    root = os.path.expanduser(a.stacks)
    for name in sorted(os.listdir(root)):
        meta_p = os.path.join(root, name, "meta.json")
        if not os.path.isfile(meta_p):
            continue
        meta = json.load(open(meta_p))
        od = os.path.join(a.out, name)
        os.makedirs(od, exist_ok=True)
        for i, s in enumerate(meta["slices"]):
            src = os.path.join(root, name, f"s{s['position']:05d}_f{a.frame:02d}.png")
            if not os.path.isfile(src):
                raise SystemExit(f"missing {src}")
            dst = os.path.join(od, f"{i:05d}.png")
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(src, dst)
        print(f"{name}: {len(meta['slices'])} frames ({meta['direction']})")


if __name__ == "__main__":
    main()
