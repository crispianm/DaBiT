<div align="center">

<h1>DaBiT: Depth and Blur informed Transformer for Video Focal Deblurring</h1>

<a href="https://arxiv.org/abs/2407.01230"><img src="https://img.shields.io/badge/arXiv-2407.01230-b31b1b.svg"></a>
<a href="https://github.com/crispianm/DaBiT"><img src="https://img.shields.io/badge/GitHub-DaBiT-blue.svg?logo=github"></a>

</div>

DaBiT restores videos degraded by **focal (defocus) blur**. It uses per-frame **depth**
(from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)) and **blur maps**
to guide a map-conditioned temporal transformer, together with optical-flow image propagation
(from [RAFT](https://github.com/princeton-vl/RAFT)). The deblurred frames are additionally
**2× super-resolved** by a cascade of PixelShuffle layers. The architecture is adapted from
[ProPainter](https://github.com/sczhou/ProPainter).

## Results

Evaluated on **DAVIS-Blur** (DAVIS-2017 with synthetic focal blur, 90 sequences). Paper numbers
are from arXiv:2407.01230v3; "Reproduced" is this repository with `weights/dabit.pth` on an
RTX 4090 (`python test_dabit.py`).

| Metric | Paper | Reproduced (this repo) |
|--------|------:|-----------------------:|
| PSNR ↑  | 28.777 | **28.480** |
| SSIM ↑  | 0.811  | 0.841 |
| tOF ↓   | 1.239  | 1.189 |
| LPIPS ↓ | –      | 0.242 |

(Mean over all 90 DAVIS-Blur sequences, `weights/dabit.pth`, RTX 4090.)

PSNR and tOF match the paper. SSIM is reported with the canonical Gaussian-window implementation
(`skimage`, `gaussian_weights=True, sigma=1.5`); small differences from the paper come from the
SSIM implementation/window. tOF (temporal optical-flow error) follows the
[FMA-Net](https://github.com/KAIST-VICLab/FMA-Net) implementation. LPIPS (AlexNet) is reported
for completeness.

Retraining from scratch with this repository (`train.py`, `configs/dabit.json`) also reproduces
the paper: a 250k-iteration run reaches **28.62 PSNR / 0.844 SSIM / 1.260 tOF / 0.240 LPIPS**
on the same benchmark.

### Improved models

Two stronger models are released beyond the paper weights (all numbers below are the mean over
all 90 DAVIS-Blur sequences, full fp32 inference):

| Model | Weights | PSNR ↑ | SSIM ↑ | LPIPS ↓ | tOF ↓ |
|-------|---------|-------:|-------:|--------:|------:|
| Paper (`dabit.pth`)                | [v1.1](https://github.com/crispianm/DaBiT/releases/tag/v1.1) | 28.48 | 0.841 | 0.242 | 1.189 |
| **Retrained, expanded data** (`dabit_retrained.pth`)   | [v1.1](https://github.com/crispianm/DaBiT/releases/tag/v1.1) | **29.17** | **0.858** | 0.217 | 1.103 |
| **Perceptual + temporal FT** (`dabit_perceptual.pth`)  | [v1.2](https://github.com/crispianm/DaBiT/releases/tag/v1.2) | 28.99 | 0.856 | **0.173** | **0.882** |

- **`dabit_retrained.pth`** — the paper architecture and recipe, retrained for 300k iterations
  on a larger clean-video corpus (see [Datasets](#datasets--data-preparation-for-training)).
  Pure fidelity gain: **+0.69 dB PSNR** over the released weights with no other change.
- **`dabit_perceptual.pth`** — `dabit_retrained.pth` fine-tuned for 50k iterations with an added
  LPIPS-VGG perceptual loss and a flow-warped, occlusion-masked temporal-consistency loss
  (`losses.perc_weight` / `losses.temp_weight` in the config). Trades ~0.18 dB PSNR for a **20%
  lower LPIPS** and **20% lower tOF** — sharper detail and smoother motion. Pick this one for
  perceptual quality, `dabit_retrained.pth` for peak PSNR/SSIM.

Evaluation is heavily optimised: ground-truth frames are prefetched in background worker
processes, Depth Anything is cached per frame, and PSNR/SSIM/tOF are computed in parallel across
CPU cores, so the full 90-sequence run is GPU-bound rather than CPU-bound. Pass `--save_videos`
to additionally write the refocused frames to `results/<sequence>/`.

## Installation

Tested with **Python 3.11** and **CUDA 12.4** on an RTX 4090.

```bash
git clone https://github.com/crispianm/DaBiT.git
cd DaBiT

# create an environment (uv shown; venv/conda also fine)
uv venv --python 3.11 .venv
source .venv/bin/activate

# one command — pulls the CUDA build of torch and all deps
pip install -r requirements.txt
```

> `pytorch_wavelets` still imports the legacy `pkg_resources` API, so `requirements.txt`
> pins `setuptools<81`. The CUDA wheel index for torch is set at the top of `requirements.txt`;
> change it for a different CUDA/CPU setup (see https://pytorch.org/get-started/locally).

## Pretrained weights

All weights are published as GitHub Release assets. The
[**v1.1** release](https://github.com/crispianm/DaBiT/releases/tag/v1.1) contains every model
plus the RAFT and Depth Anything V2 dependencies needed to run anything in this repo; the
[**v1.2** release](https://github.com/crispianm/DaBiT/releases/tag/v1.2) adds the perceptual
model. Download them into `weights/`:

```bash
mkdir -p weights && cd weights
# core deps + models (needed to run test_dabit.py / train.py)
gh release download v1.1 -R crispianm/DaBiT      # or download from the release page
gh release download v1.2 -R crispianm/DaBiT      # perceptual model (optional)
cd ..
```

```
weights/
   |- dabit.pth                     # paper model (~182 MB)
   |- dabit_retrained.pth           # retrained on expanded data — best PSNR/SSIM
   |- dabit_perceptual.pth          # perceptual + temporal fine-tune — best LPIPS/tOF
   |- raft.pth                      # RAFT optical flow (~21 MB)
   |- depth_anything_v2_vits.pth    # Depth Anything V2, ViT-S (~95 MB) — used at train/test time
   |- depth_anything_v2_vitb.pth    # (optional) ViT-B — for get_depths.py
   |- depth_anything_v2_vitl.pth    # (optional) ViT-L — get_depths.py default
```

Evaluation needs `dabit*.pth` (choose one), `raft.pth`, and `depth_anything_v2_vits.pth`. The
RAFT and Depth Anything V2 weights are redistributed unchanged from their original releases
([RAFT](https://github.com/princeton-vl/RAFT), BSD-3; [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
Apache-2.0) for convenience.

## Reproduce the paper results (evaluation)

The DAVIS-Blur test set ships in [`blur_tests/`](blur_tests) (90 sequences, each with
`frames/` blurred input, `gt/` sharp ground truth, `masks/` blur maps and `depths/`), so
evaluation is **self-contained** — no external dataset is required.

```bash
python test_dabit.py                                       # paper weights (weights/dabit.pth)
python test_dabit.py --model weights/dabit_retrained.pth   # best PSNR/SSIM
python test_dabit.py --model weights/dabit_perceptual.pth  # best LPIPS/tOF
```

It runs RAFT → Depth Anything V2 → DaBiT over every sequence and prints the per-sequence and
mean PSNR / SSIM / LPIPS / tOF (logged to `results/results.txt`). Add `--save_videos` to also
write the refocused/upscaled frames to `results/<sequence>/`.

## Datasets & data preparation (for training)

The paper model is trained on **YouTube-VOS** (3,471 clips) + **BVI-DVC** (200 clips), with focal
blur synthesised on the fly. `dabit_retrained.pth` additionally uses **TartanAir-V2** (561 rendered
camera trajectories) and **Virtual KITTI 2** (100 sequences) as extra clean-video sources — a
~1.4 M-frame corpus in total. Because the pipeline only needs *clean* video (blur, blur maps, and
depth are all synthesised/estimated), any diverse sharp-video collection can be added the same way:
just point a dataset entry at it. **Ground-truth depth is never used** — depths for all datasets are
estimated with Depth Anything V2 (`get_depths.py`), so the depth distribution driving blur synthesis
matches what the model sees at inference. The synthetic-blur test set (DAVIS-Blur) is built from
DAVIS-2017 using the per-sequence parameters in [`davis_blur.csv`](davis_blur.csv).

| Script | Purpose |
|--------|---------|
| [`get_depths.py`](get_depths.py) | Pre-compute Depth Anything V2 depth maps for a frame folder |
| [`get_blur_maps.py`](get_blur_maps.py) | Generate per-frame focal-blur maps |
| [`create_davis_blur.py`](create_davis_blur.py) | Build the DAVIS-Blur test set from DAVIS + `davis_blur.csv` |

`core/dataset.py:TrainDataset` reads `frames/<seq>/...` and `depths/<seq>/...` under each
dataset's `video_root` by default; datasets whose frames and depths live in separate trees
(e.g. YouTube-VOS) can override either path with the `frames_root`/`depths_root` config keys —
no symlinks or copies needed. Depth maps are accepted in any common format (8/16-bit
PNG/JPG/TIFF, or raw `.npy`/`.npz` arrays), matched to frames by name (`<frame>_depth.png`,
`<frame>.npz`, ...) or, failing that, by sorted index. Paths are set in
[`configs/dabit.json`](configs/dabit.json), e.g.:

```
<your-data-root>/
   |- bvidvc/        {frames,depths}/<seq>/...        # default self-contained layout
   |- youtube-vos/   rgb/<seq>/...                    # frames_root
   |- youtube-vos-depth/ <seq>/...                    # depths_root (estimated by get_depths.py)
```

Edit the dataset paths in the config to point at your data root before training.

> **Warning:** do **not** train on DAVIS — DAVIS-Blur (the test set) is built from the same
> sequences, so adding it to training leaks the benchmark.

## Training

```bash
# single GPU
python train.py -c configs/dabit.json

# multi-GPU (data-parallel); scale the config's batch_size / lr accordingly
torchrun --standalone --nproc_per_node=<N> train.py -c configs/dabit.json
```

`configs/dabit.json` follows the paper recipe: lr 1e-4, batch size 1, **300k iterations**,
432×240 inputs (864×480 ground truth for the 2× super-resolution head), 10 local + 6 reference
frames, focal blur synthesised on the fly from the depth maps. Checkpoints and TensorBoard logs
are written to `out_dir`; a small DAVIS-Blur subset is validated every `val_freq` iterations, and
training auto-resumes from the latest checkpoint in `out_dir` on re-launch.

Other configs are provided as examples: `configs/dabit_isambard.json` (the expanded-data recipe
behind `dabit_retrained.pth`) and `configs/dabit_ft.json` (perceptual + temporal fine-tuning
behind `dabit_perceptual.pth`; set `losses.perc_weight` / `losses.temp_weight` and a `model_path`
to fine-tune from). The `jobs_*.sh` scripts are the SLURM wrappers used on the Isambard-AI
cluster and are provided as references (they contain cluster-specific paths).

For robustness the trainer deviates from the paper in a few documented ways: AdamW (instead of
Adam) with cosine LR decay + linear warmup, bf16 mixed precision, gradient clipping and
non-finite loss/grad guards, plus optional multi-GPU (DDP) and EMA weights. Set `trainer.amp` to
`false` and `trainer.scheduler` accordingly to get closer to the original fixed-LR fp32 recipe.
(Pretrained weights are provided, so retraining is not required to reproduce the results.)

## Model

`model/dabit.py` defines the **`DaBiT`** network (45.3 M parameters): a 6-channel encoder
(RGB + masks + depth), learnable bidirectional flow propagation, an 8-layer map-guided
temporal sparse transformer, a decoder, and three cascading PixelShuffle layers for 2×
super-resolution.

## Repository layout

```
test_dabit.py            # evaluation / paper reproduction
train.py                 # training entry point
configs/dabit.json       # training configuration
get_depths.py            # depth-map precomputation
get_blur_maps.py         # blur-map generation
create_davis_blur.py     # build the DAVIS-Blur test set
davis_blur.csv           # per-sequence blur parameters
blur_tests/              # DAVIS-Blur test set (self-contained)
core/                    # dataset, trainer, losses, metrics, utils
model/                   # DaBiT + super-resolution backbone, modules/ (transformer, depth, deform-conv)
RAFT/                    # optical flow
weights/                 # pretrained models (downloaded separately)
```

## Acknowledgement

Built on [ProPainter](https://github.com/sczhou/ProPainter),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[RAFT](https://github.com/princeton-vl/RAFT). Thanks to the authors for releasing their code.

## License

S-Lab License 1.0 (non-commercial). See [LICENSE](LICENSE).

## Citation

```bibtex
@article{morris2024dabit,
  title   = {DaBiT: Depth and Blur informed Transformer for Video Focal Deblurring},
  author  = {Morris, Crispian and others},
  journal = {arXiv preprint arXiv:2407.01230},
  year    = {2024}
}
```
