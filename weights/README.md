# Weights

Download the pretrained models from the GitHub Release assets into this folder:

```bash
gh release download v1.1 -R crispianm/DaBiT   # models + RAFT/Depth Anything deps
gh release download v1.2 -R crispianm/DaBiT   # perceptual model (optional)
```

Evaluation needs one `dabit*.pth` model, `raft.pth`, and `depth_anything_v2_vits.pth`.

```
weights
   |- dabit.pth                     # paper model (~182 MB)
   |- dabit_retrained.pth           # retrained on expanded data — best PSNR/SSIM
   |- dabit_perceptual.pth          # perceptual + temporal fine-tune — best LPIPS/tOF
   |- raft.pth                      # RAFT optical flow (~21 MB)
   |- depth_anything_v2_vits.pth    # Depth Anything V2, ViT-S (~95 MB) — train/test time
   |- depth_anything_v2_vitb.pth    # (optional) ViT-B, for get_depths.py
   |- depth_anything_v2_vitl.pth    # (optional) ViT-L, for get_depths.py
   |- README.md
```
