import sys
import os
from vot_path import base_path
sys.path.insert(0, os.path.join(base_path, 'DiMP_LTMU/Global_Track/_submodules/neuron'))
sys.path.insert(0, os.path.join(base_path, 'DiMP_LTMU/Global_Track/_submodules/mmdetection'))
sys.path.insert(0, os.path.join(base_path, 'DiMP_LTMU/Global_Track'))
# register modules
from modules import *
from datasets import *
