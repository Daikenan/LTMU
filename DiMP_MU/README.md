
If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Introduction

This is a python-implemented visual object tracking algorithm. Use meta-updater to control the update of DiMP.

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
source create -n DiMP_MU python=3.7
source activate DiMP_MU
cd path/to/DiMP_MU
pip install -r requirements.txt
```
4.Download models
Download [[metric model](https://drive.google.com/open?id=1o-btxlWWA6GlbwMGCGkzn2vAw9qv8D2z)] [[DiMP50 model](https://drive.google.com/file/d/1qgachgqks2UGjKx-GdO1qylBDdB1f9KN/view)]and put in the following path:

```
 utils/metric_net/metric_model/metric_model.pt
 DiMP_MU/pytracking/networks/dimp50.pth
 ```
5. Run the demo script to test the tracker:
```
cd path/to/DiMP_MU
source activate DiMP_MU
python Demo.py
```
# Training tutorial
Refer to [ATOM_MU](https://github.com/Daikenan/LTMU/tree/master/ATOM_MU).
