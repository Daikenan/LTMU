
If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Introduction

This is a python-implemented visual object tracking algorithm. 

# Prerequisites

* python 3.7
* ubuntu 16.04
* cuda-10.0

## Install Requirements & 
To install all the dependencies, you can run:
``
source create -n LTMU_B python=3.7
source activate LTMU_B
cd path/to/DiMP_LTMU
pip install -r requirements.txt
`` 
## Download models and put in the following path
https://drive.google.com/open?id=1_IOhsY4SJPQvEhbPP1sttGHpn81vfjIW

`DiMP_LTMU/Global_Track/checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth`
`DiMP_LTMU/pyMDNet/models/mdnet_imagenet_vid.pth`
 `utils/metric_net/metric_model/metric_model.pt`
`DiMP_LTMU/SiamMask/experiments/siammask/SiamMask_VOT_LD.pth`

compile the Cpp/CUDA extensions by running `python setup.py develop` under `DiMP_LTMU/Global_Track/_submodules/mmdetection`.

## Modify the Paths
* `vot_path.py`

