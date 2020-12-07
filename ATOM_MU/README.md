
If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Introduction

This is a python-implemented visual object tracking algorithm. Use meta-updater to control the update of ATOM.

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
Download [[metric model](https://drive.google.com/open?id=1o-btxlWWA6GlbwMGCGkzn2vAw9qv8D2z)] [[ATOM model](https://drive.google.com/open?id=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU)] and put in the following path:

```
utils/metric_net/metric_model/metric_model.pt
ATOM_MU/pytracking/networks/atom_default.pth
```
5. Run the demo script to test the tracker:
```
cd path/to/ATOM_MU
source activate DiMP_MU
python Demo.py
```

# Training tutorial
1. Run original tracker and record all results(bbox, response map,...) on LaSOT dateset to get first round training data of meta-updater.You can modify test_tracker.py like this:
```
import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
p = p_config()
p.save_training_data = True
p.tracker = 'ATOM'
p.name = 'atom_test'
eval_tracking('lasot', p=p, mode='all')
```

2. In `meta_updater/tcopt.py`, modify `tcopts['lstm_train_dir']` and `tcopts['train_data_dir']` like this:
```
tcopts['lstm_train_dir'] = './atom_mu_test_1'
tcopts['train_data_dir'] = '../results/atom_test/lasot/train_data'
```
`tcopts['train_data_dir']` is dir of the training data and `tcopts['lstm_train_dir']` is the save path of training models.

3. Run `meta_updater/train_meta_updater` to train meta-updater.After training, you can evaluate your own meta-updater by modifying `test_tracker.py` like this:
```
import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
p = p_config()
p.save_training_data = True
p.tracker = 'ATOM_MU'
p.name = 'atom_mu_test_1'
eval_tracking('lasot', p=p, mode='test')
p.save_training_data = False
eval_tracking('tlp', p=p)
eval_tracking('votlt19', p=p)
```
And then run `evaluate.results.py` to evaluate results.

For most trackers, one stage of training can achieve a significant performance improvement. If you want to get better performance, you can try multiple stages of training like this:

4. Run tracker with meta-updater and record all results(bbox, response map,...) on LaSOT dateset to get first round training data of meta-updater.You can modify test_tracker.py like this:
```
import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
p = p_config()
p.save_training_data = True
p.tracker = 'ATOM_MU'
p.name = 'atom_mu_test_1'
eval_tracking('lasot', p=p, mode='all')
```

5. In `meta_updater/tcopt.py`, modify `tcopts['lstm_train_dir']` and `tcopts['train_data_dir']` like this:
```
tcopts['lstm_train_dir'] = './atom_mu_test_2'
tcopts['train_data_dir'] = '../results/'atom_mu_test_1'/lasot/train_data'
```
`tcopts['train_data_dir']` is dir of the training data and `tcopts['lstm_train_dir']` is the save path of training models.

6. Run `meta_updater/train_meta_updater` to train meta-updater.After training, you can evaluate your own meta-updater by modifying `test_tracker.py` like this:
```
import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
p = p_config()
p.save_training_data = True
p.tracker = 'ATOM_MU'
p.name = 'atom_mu_test_2'
eval_tracking('lasot', p=p, mode='test')
p.save_training_data = False
eval_tracking('tlp', p=p)
eval_tracking('votlt19', p=p)
```
And then run `evaluate.results.py` to evaluate results. 

Modify the corresponding `p.name`,`tcopts['lstm_train_dir']`,`tcopts['train_data_dir']` and repeat 4-6;
