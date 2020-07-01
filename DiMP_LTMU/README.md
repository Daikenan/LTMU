
If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Introduction

This is a python-implemented visual object tracking algorithm. 

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
You can also download models manually and put in the following path:
```
`DiMP_LTMU/Global_Track/checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth`
`DiMP_LTMU/pyMDNet/models/mdnet_imagenet_vid.pth`
 `utils/metric_net/metric_model/metric_model.pt`
`DiMP_LTMU/SiamMask/experiments/siammask/SiamMask_VOT_LD.pth`
```
5. modify ``vot_path.py``:
6. Run the demo script to test the tracker:
```
cd path/to/DiMP_LTMU
source activate DiMP_LTMU
python Demo.py
```

