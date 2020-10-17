
If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Introduction

This is a python-implemented visual object tracking algorithm. 

# Prerequisites

* python 3.6
* ubuntu 16.04
* cuda-9.0

## Install Requirements & Download models
To install all the dependencies, you can run the script `install.sh`. 
Usage example:
``
bash install.sh ~/anaconda3 votenvs
``
The first parameter `~/anaconda3` indicates the path of anaconda and the second indicates the virtual environment used for this project. 

## Modify the Paths
* `local_path.py`

# Integrate into VOT-2019LT

## VOT-toolkit
Before running the toolkit, please change the environment path to use the python in the conda environment "votenvs".
For example, in my computer, I add  `export PATH=/home/daikenan/anaconda3/envs/votenvs/bin:$PATH` to the `~/.bashrc` file.  

The interface for integrating the tracker into the vot evaluation tool kit is implemented in the module `tracker_vot.py`. The script `tracker_LT_DSE.m` is needed to be copied to vot-tookit. 

Since the vot-toolkit may be not compatible with pytorch-0.4.1, I always change the line  `command = sprintf('%s %s -c "%s"', python_executable, argument_string, python_script);` to `command = sprintf('env -i %s %s -c "%s"', python_executable, argument_string, python_script);` in `generate_python_command.m`. 


# CPU manner

If you want to run this code on CPU, you need just set `os.environ["CUDA_VISIBLE_DEVICES"]=""` in the begin of `tracker_vot.py`. 
