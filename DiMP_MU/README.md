
If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Introduction

This is a python-implemented visual object tracking algorithm.Use meta-updater to control the update of Dimp.

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
Download [models](https://drive.google.com/open?id=1o-btxlWWA6GlbwMGCGkzn2vAw9qv8D2z) and put in the following path:

```
 utils/metric_net/metric_model/metric_model.pt
 ```
5. Run the demo script to test the tracker:
```
cd path/to/DiMP_MU
source activate DiMP_MU
python Demo.py
```
<<<<<<< HEAD

=======
>>>>>>> 39ec09e14eb2ad96b4356830fd6108b70a808f6c


