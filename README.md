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

Place the following in `weights/` (evaluation needs the first three):

```
weights/
   |- dabit.pth                     # trained DaBiT model (~182 MB)
   |- raft.pth                      # RAFT optical flow (~21 MB)
   |- depth_anything_v2_vits.pth    # Depth Anything V2, ViT-S (~95 MB)
   |- depth_anything_v2_vitb.pth    # (optional) ViT-B
   |- depth_anything_v2_vitl.pth    # (optional) ViT-L
```

`dabit.pth` is the final model. `dabit_280.pth` is the 280k-iteration checkpoint
(`output_step_lr/model_280000.pth`). RAFT and Depth Anything V2 weights are from their
respective releases. **Download the weights from the GitHub Release assets** (or the link in
the repo description) and unzip into `weights/`.

## Reproduce the paper results (evaluation)

The DAVIS-Blur test set ships in [`blur_tests/`](blur_tests) (90 sequences, each with
`frames/` blurred input, `gt/` sharp ground truth, `masks/` blur maps and `depths/`), so
evaluation is **self-contained** — no external dataset is required.

```bash
python test_dabit.py                     # uses weights/dabit.pth, writes results/ + results.txt
python test_dabit.py --model weights/dabit_280.pth   # evaluate a different checkpoint
```

It runs RAFT → Depth Anything V2 → DaBiT over every sequence and prints the per-sequence and
mean PSNR / SSIM / LPIPS / tOF (logged to `results/results.txt`). Add `--save_videos` to also
write the refocused/upscaled frames to `results/<sequence>/`.

## Datasets & data preparation (for training)

DaBiT is trained on **YouTube-VOS** (3,471 clips) + **BVI-DVC** (200 clips), with focal blur
synthesised on the fly. The synthetic-blur test set (DAVIS-Blur) is built from DAVIS-2017 using
the per-sequence parameters in [`davis_blur.csv`](davis_blur.csv).

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
/mnt/DATA/dabit_training_data/
   |- bvidvc/        {frames,depths}/<seq>/...                # default layout
   |- youtube-vos/   train_all_frames/JPEGImages/<seq>/...    # frames_root
   |- youtube-vos-depth/ vos-output/train_all_frames/depth/   # depths_root (.npz)
```

> **Warning:** do **not** train on DAVIS — DAVIS-Blur (the test set) is built from the same
> sequences, so adding it to training leaks the benchmark.

## Training

```bash
python train.py -c configs/dabit.json
```

`configs/dabit.json` follows the paper recipe: lr 1e-4, batch size 1, **300k iterations**,
432×240 inputs (864×480 ground truth for the 2× super-resolution head), 10 local + 6 reference
frames, focal blur synthesised on the fly from the depth maps. Checkpoints and TensorBoard logs
are written to `out_dir`; a small DAVIS-Blur subset is validated every `val_freq` iterations.

For robustness the released trainer deviates from the paper in a few documented ways: AdamW
(instead of Adam) with cosine LR decay + linear warmup, bf16 mixed precision, gradient
clipping and non-finite loss/grad guards. Set `trainer.amp` to `false` and
`trainer.scheduler` accordingly to get closer to the original fixed-LR fp32 recipe. (The
released `weights/dabit.pth` is provided, so retraining is not required to reproduce the
results.)

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
