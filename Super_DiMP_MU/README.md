
If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Introduction

This is a python-implemented visual object tracking algorithm. Use meta-updater to control the update of Super DiMP.

# Prerequisites

* python 3.7
* ubuntu 16.04
* cuda-10.0

## Install Requirements & 
To install all the dependencies, you can run:

`
source create -n DiMP_MU python=3.7
`

`
source activate DiMP_MU
`

`
cd path/to/Super_DiMP_MU
`
`
pip install -r requirements.txt
`
## Download models and put in the following path
https://drive.google.com/open?id=1o-btxlWWA6GlbwMGCGkzn2vAw9qv8D2z

 `utils/metric_net/metric_model/metric_model.pt`


# Training tutorial
Refer to [ATOM_MU](https://github.com/Daikenan/LTMU/tree/master/ATOM_MU).
