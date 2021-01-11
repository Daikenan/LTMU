
If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Introduction

The proposed algorithm consists of four modules, e.g. local tracker, verifier, global detector and meta-updater. The presence of the target is judged by the outputs from both the local tracker and the verifier. When disappearance is detected, the tracker triggers the global detector to conduct image-level detection and obtain candidate proposals. If the verifier finds the target among the proposals, then the local tracker will be reset based on the current tracking result. The update of the local tracker is controlled by the meta-updater.

The short-term local tracker contains two components. One is for target localization and based on DiMP algorithm[1]. It uses ResNet50 as the backbone network. The input of it is the local search region and it outputs a single response map, in which the center of the target has the highest response. The other component is the SiamMask network[2] and used for refining the bounding box after locating the center of the target. It also takes the local search region as the input and outputs the tight bounding boxes of candidate proposals. 

For the verifier, we adopts MDNet network[3] which uses VGGM as the backbone and is pre-trained on ILSVRC VID dataset. The input of it is the local search region as well and in each frame, we crop the feature of the search region outputted by the third convolutional layer to get the feature of the tracking result. The classification score is finally obtained by sending the tracking result's feature to three fully connected layers.
Meta_Updater[4] composed of LSTMs that takes historical bounding box data, appearance cues and discriminative features into account towards deciding whether or not the model should be updated. It takes historical bounding box ,tracker scores, response map and image as the input and outs a binary classification, whether the current frame should be updated. 

We utilize GlobalTrack[5] as the global detector, which is a one-shot detection algorithm and can search the candidates in the whole image.

This method is a simplified version of LTMU and LTDSE with comparable performance, which additionally has a RPN-based regression network, a sliding-window based re-detection module and a complex mechanism for updating models and target re-localization.

# Prerequisites

* python 3.7
* ubuntu 16.04
* cuda-10.0

# Installation
1. Clone the GIT repository:
```
 $ git clone https://github.com/Daikenan/LTMU.git
```
2. Clone the submodules.  
   In the repository directory, run the commands:
```
   $ git submodule init  
   $ git submodule update
```
3. Run the install script. 
```
cd path/to/DiMP_LTMU
source create -n DiMP_LTMU python=3.7
source activate DiMP_LTMU
pip install -r requirements.txt
bash compile.sh
```
4.Download models
```
bash download_models.sh
```
You can also download models([[Google drive](https://drive.google.com/open?id=1_IOhsY4SJPQvEhbPP1sttGHpn81vfjIW)], [[Baidu yun(extract code: gzjm)](https://pan.baidu.com/s/1-ZReq_Ls63IqsSQ28rdTXA)]) manually and put in the following path:
```
DiMP_LTMU/Global_Track/checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth
DiMP_LTMU/pyMDNet/models/mdnet_imagenet_vid.pth
utils/metric_net/metric_model/metric_model.pt
DiMP_LTMU/SiamMask/experiments/siammask/SiamMask_VOT_LD.pth
```
5. modify ``vot_path.py``:
6. Run the demo script to test the tracker:
```
cd path/to/DiMP_LTMU
source activate DiMP_LTMU
python Demo.py
```
# Performance
update soon
# Citation
```
[1]@inproceedings{Danelljan-ICCV19-DIMP,
  author    = {Goutam Bhat and
                     Martin Danelljan and
                     Luc Van Gool and
                     Radu Timofte},
  title        = {Learning Discriminative Model Prediction for Tracking},
  booktitle = {ICCV},
  year        = {2019},
}
[2] @inproceedings{wang2019fast,
  title={Fast online object tracking and segmentation: A unifying approach},
  author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
  booktitle={CVPR},
  pages={1328--1338},
  year={2019}
}
[3]@InProceedings{nam2016mdnet,
author = {Nam, Hyeonseob and Han, Bohyung},
title = {Learning Multi-Domain Convolutional Neural Networks for Visual Tracking},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2016}
}
[4]@inproceedings{Dai_2020_CVPR,
author = {Kenan Dai, Yunhua Zhang, Dong Wang, Jianhua Li, Huchuan Lu, Xiaoyun Yang},
title = {{High-Performance Long-Term Tracking with Meta-Updater},
booktitle = {CVPR},
year = {2020}
}
[5]@inproceedings{GlobalTrack,
  author    = {Lianghua Huang and Xin Zhao and Kaiqi Huang},
  title        = {{GlobalTrack: A} Simple and Strong Baseline for Long-term Tracking},
  booktitle = {AAAI},
  year        = {2020},
}
