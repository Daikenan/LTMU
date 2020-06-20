# GlobalTrack

> UPDATES:<br>
> - [2020.03.02] Update training scripts to match the settings in the paper (12 epochs on `COCO` and another 12 epochs on `COCO + GOT + LaSOT`)!
> - [2020.02.19] Both training and evaluation code are available!<br>
> - [2020.02.19] Initial and pretrained weights are provided!<br>
> - [2020.02.19] A demo tracking video of GlobalTrack is available [here](https://youtu.be/na0H3u4cLqY)!

Official implementation of our AAAI2020 paper: GlobalTrack: A Simple and Strong Baseline for Long-term Tracking. **The first tracker with NO cumulative errors.**

![figure2](imgs/figure2.jpg)

Extremely simple tracking process, with **NO motion model, NO online learning, NO punishment on position or scale changes, NO scale smoothing and NO trajectory refinement**.

Outperforms [SPLT](https://github.com/iiau-tracker/SPLT) (ICCV19), [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf), [ATOM](https://github.com/visionml/pytracking) and [MBMD](https://github.com/xiaobai1217/MBMD) on [TLP](https://amoudgl.github.io/tlp/) benchmark (avg. **13,529 frames** per video) by **MORE THAN 11%** (absolute gain).

Outperforms [SPLT](https://github.com/iiau-tracker/SPLT), [SiamRPN++](https://github.com/STVIR/pysot), [ATOM](https://github.com/visionml/pytracking) and [DaSiamLT](https://github.com/foolwood/DaSiamRPN) on [LaSOT](https://cis.temple.edu/lasot/) benchmark.

Paper on arXiv: [1912.08531](https://arxiv.org/abs/1912.08531).

Demo video: [YouTube](https://youtu.be/na0H3u4cLqY), [YouKu](https://v.youku.com/v_show/id_XNDU0OTc5MTg3Ng==.html).

## Installation

To reproduce our Python environment, you'll need to create a conda environment from `environment.yml` and compile the Cpp/CUDA extensions (we use `CUDA toolkit 9.0`):

```shell
conda env create -f environment.yml
conda activate GlobalTrack
git clone https://github.com/huanglianghua/GlobalTrack.git
cd _submodules/mmdetection
python setup.py develop
```

Alternatively, you can also install `PyTorch==1.1.0, torchvision, shapely` and `scipy` manually, then compile the Cpp/CUDA extensions by running `python setup.py develop` under `_submodules/mmdetection`.

## Run Training

(Assuming all datasets are stored in `~/data`) Distributed training:

```shell
sh tools/dist_train_qg_rcnn.sh
```

Non-distributed training:

```
python tools/train_qg_rcnn.py --config configs/qg_rcnn_r50_fpn.py --load_from checkpoints/qg_rcnn_r50_fpn_2x_20181010-443129e1.pth --gpus 1
```

Before train, you'll need to download the [initial weights](https://drive.google.com/open?id=1JkOqbSQJvGiGb9ubFIu5M84jeTInFJTX) transferred from FasterRCNN (provided by [mmdetection](https://github.com/open-mmlab/mmdetection), pretrained on COCO) to start.

Change the arguments in `dist_train_qg_rcnn.sh` or append them to `python tools/train_qg_rcnn.py` for your need. See `train_qg_rcnn.py` for details.

## Run Tracking

(Assuming all datasets are stored in `~/data`).

```shell
python tools/test_global_track.py
```

Change the parameters, such as `cfg_file`, `ckp_file` and `evaluators` in `test_global_track.py` for your need.

## Pretrained Weights

- Initial weights transferred from FasterRCNN:
    - Google Drive: https://drive.google.com/open?id=1JkOqbSQJvGiGb9ubFIu5M84jeTInFJTX
    - Baidu Yun: [link] https://pan.baidu.com/s/1roviZxaV-A4QpS7FFSYT0g  [password] 38qo

- Pretrained GlobalTrack:
    - Google Drive: https://drive.google.com/open?id=1ouTJDS5NACLukd810uiKCvmLxJfga48x
    - Baidu Yun: [link] https://pan.baidu.com/s/19Z7vWceXeF1EEpsOW_V-dQ  [password] 47p4

By defaults, all pretrained weights are saved at `checkpoints`.

## Issues

Please report issues in this repo if you have any problems.
