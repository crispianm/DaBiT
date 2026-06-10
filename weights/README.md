# Weights

Put the downloaded pretrained models in this folder (see the GitHub Release
assets). Evaluation needs the first three:

```
weights
   |- dabit.pth                     # trained DaBiT model (~182 MB)
   |- raft.pth                      # RAFT optical flow (~21 MB)
   |- depth_anything_v2_vits.pth    # Depth Anything V2, ViT-S (~95 MB)
   |- dabit_280.pth                 # (optional) 280k-iteration checkpoint
   |- depth_anything_v2_vitb.pth    # (optional) ViT-B, for get_depths.py
   |- depth_anything_v2_vitl.pth    # (optional) ViT-L, for get_depths.py
   |- README.md
```
