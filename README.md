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
| PSNR ↑ | 28.777 | **28.480** |
| SSIM ↑ | 0.811 | 0.841 |
| LPIPS ↓ | – | 0.242 |

(Mean over all 90 DAVIS-Blur sequences, `weights/dabit.pth`, RTX 4090.)

PSNR matches the paper. SSIM is reported with the canonical Gaussian-window implementation
(`skimage`, `gaussian_weights=True, sigma=1.5`); small differences from the paper come from the
SSIM implementation/window. LPIPS (AlexNet) is reported for completeness.

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

It runs RAFT → Depth Anything V2 → DaBiT over every sequence, saves the refocused/upscaled
frames to `results/<sequence>/`, and prints the per-sequence and mean PSNR / SSIM / LPIPS.

## Datasets & data preparation (for training)

DaBiT is trained on **YouTube-VOS** (3,471 clips) + **BVI-DVC** (200 clips), with focal blur
synthesised on the fly. The synthetic-blur test set (DAVIS-Blur) is built from DAVIS-2017 using
the per-sequence parameters in [`davis_blur.csv`](davis_blur.csv).

| Script | Purpose |
|--------|---------|
| [`get_depths.py`](get_depths.py) | Pre-compute Depth Anything V2 depth maps for a frame folder |
| [`get_blur_maps.py`](get_blur_maps.py) | Generate per-frame focal-blur maps |
| [`create_davis_blur.py`](create_davis_blur.py) | Build the DAVIS-Blur test set from DAVIS + `davis_blur.csv` |

`core/dataset.py:TrainDataset` expects each dataset's `video_root` to contain `frames/<seq>/...`
and `depths/<seq>/...`. On the training machine the datasets live under
`/mnt/DATA/dabit_training_data/` and the paths are set in [`configs/dabit.json`](configs/dabit.json):

```
/mnt/DATA/dabit_training_data/
   |- bvidvc/        {frames,depths}/<seq>/...          # matches TrainDataset directly
   |- davis/         {frames,depths}/<seq>/...
   |- youtube-vos/   train_all_frames/JPEGImages/<seq>/...   # frames
   |- youtube-vos-depth/ vos-output/train_all_frames/depth/  # depths (separate tree)
```

> **Note:** YouTube-VOS frames and depths live in separate trees, so its `video_root` needs
> `frames/` and `depths/` symlinks (or a copy) laid out as above before training. BVI-DVC and
> DAVIS already match the expected layout.

## Training

```bash
python train.py -c configs/dabit.json
```

Key settings in `configs/dabit.json`: AdamW, lr 1e-4, batch size 1, 750k iterations,
CosineAnnealingLR, 432×240 crops, 10 local + 6 reference frames. Checkpoints are written to
`out_dir`. (The released `weights/dabit.pth` is provided, so retraining is not required to
reproduce the results.)

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
