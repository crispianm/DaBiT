<div align="center">


<h1>DepthPainter: Depth-Aware Propagation and Transformer for Video Inpainting</h1>


<!-- <div>
    <h4 align="center">
        <a href="https://shangchenzhou.com/projects/ProPainter" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2309.03897" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2309.03897-b31b1b.svg">
        </a>
        <a href="https://youtu.be/92EHfgCO5-Q" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
        <a href="https://huggingface.co/spaces/sczhou/ProPainter" target='_blank'>
        <img src="https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-blue">
        </a>
        <a href="https://openxlab.org.cn/apps/detail/ShangchenZhou/ProPainter" target='_blank'>
        <img src="https://img.shields.io/badge/Demo-%F0%9F%91%A8%E2%80%8D%F0%9F%8E%A8%20OpenXLab-blue">
        </a>
        <img src="https://api.infinitescript.com/badgen/count?name=sczhou/ProPainter">
    </h4>
</div> -->


</div>


  




## Dependencies and Installation

1. Clone Repo

   ```bash
   git clone https://github.com/crispianm/DepthPainter.git
   ```

2. Create Conda Environment and Install Dependencies

   ```bash
   # create new anaconda env
   conda create -n depthpainter python=3.8 -y
   conda activate depthpainter

   # install python dependencies
   pip3 install -r requirements.txt
   ```

   - CUDA >= 9.2
   - PyTorch >= 1.7.1
   - Torchvision >= 0.8.2
   - Other required packages in `requirements.txt`

## Get Started
### Prepare pretrained models


The directory structure will be arranged as:
```
weights
   |- ProPainter.pth
   |- recurrent_flow_completion.pth
   |- raft-things.pth
   |- i3d_rgb_imagenet.pt (for evaluating VFID metric)
   |- README.md
```


The results will be saved in the `results` folder.
To test your own videos, please prepare the input `mp4 video` (or `split frames`) and `frame-wise mask(s)`.

If you want to specify the video resolution for processing or avoid running out of memory, you can set the video size of `--width` and `--height`:
```shell
# process a 576x320 video; set --fp16 to use fp16 (half precision) during inference.
python inference_propainter.py --video inputs/video_completion/running_car.mp4 --mask inputs/video_completion/mask_square.png --height 320 --width 576 --fp16
```



<!-- ## Dataset preparation
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>YouTube-VOS</th>
    <th>DAVIS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Description</td>
    <td>For training (3,471) and evaluation (508)</td>
    <td>For evaluation (50 in 90)</td>
  <tr>
    <td>Images</td>
    <td> [<a href="https://competitions.codalab.org/competitions/19544#participate-get-data">Official Link</a>] (Download train and test all frames) </td>
    <td> [<a href="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip">Official Link</a>] (2017, 480p, TrainVal) </td>
  </tr>
  <tr>
    <td>Masks</td>
    <td colspan="2"> [<a href="https://drive.google.com/file/d/1dFTneS_zaJAHjglxU10gYzr1-xALgHa4/view?usp=sharing">Google Drive</a>] [<a href="https://pan.baidu.com/s/1JC-UKmlQfjhVtD81196cxA?pwd=87e3">Baidu Disk</a>] (For reproducing paper results; provided in <a href="https://arxiv.org/abs/2309.03897">ProPainter</a> paper) </td>
  </tr>
</tbody>
</table> -->

The training and test split files are provided in `datasets/<dataset_name>`. For each dataset, you should place `JPEGImages` to `datasets/<dataset_name>`. Resize all video frames to size `432x240` for training. Unzip downloaded mask files to `datasets`.

The `datasets` directory structure will be arranged as: (**Note**: please check it carefully)
```
datasets
   |- davis
      |- JPEGImages_432_240
         |- <video_name>
            |- 00000.jpg
            |- 00001.jpg
      |- test_masks
         |- <video_name>
            |- 00000.png
            |- 00001.png   
      |- train.json
      |- test.json
   |- youtube-vos
      |- JPEGImages_432_240
         |- <video_name>
            |- 00000.jpg
            |- 00001.jpg
      |- test_masks
         |- <video_name>
            |- 00000.png
            |- 00001.png
      |- train.json
      |- test.json   
```




## License

This project is licensed under NTU S-Lab License 1.0. Redistribution and use should follow this license.


## Acknowledgement

This code is heavily based on [ProPainter](https://github.com/sczhou/ProPainter), thanks to the authors for providing their code!

