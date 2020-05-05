import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from me_sample_generator import *
from PIL import Image
import torch.utils.data as torch_dataset
from metric_model import ft_net
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable
from torchvision import transforms
# from tensorboardX import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path = '/home/dkn/daikenan/dataset/VOT2019/lt2019/bike1/color/'
img_list = os.listdir(data_path)
img_list.sort()
gt = np.loadtxt("/home/dkn/daikenan/dataset/VOT2019/lt2019/bike1/groundtruth.txt", delimiter=',')

img_num = len(img_list)
anchor_index = np.random.randint(low=0, high=img_num, size=[1])[0]
pos_index = np.random.randint(low=0,high=img_num, size=[1])[0]

anchor_box = gt[anchor_index,:]
anchor_box = anchor_box.reshape((1,4))
anchor_im = np.array(Image.open(data_path + img_list[anchor_index]))

pos_box = gt[pos_index,:]
pos_box = pos_box.reshape((1,4))
pos_im = Image.open(data_path + img_list[pos_index])

def process_regions(regions):
    # regions = np.squeeze(regions, axis=0)
    regions = regions[0] / 255.0
    regions[:, :, 0] = (regions[:, :, 0] - 0.485) / 0.229
    regions[:, :, 1] = (regions[:, :, 1] - 0.456) / 0.224
    regions[:, :, 2] = (regions[:, :, 2] - 0.406) / 0.225
    regions = np.transpose(regions, (2, 0, 1))
    regions = np.expand_dims(regions, axis=0)
    #regions = np.tile(regions, (2,1,1,1))
    return regions

def get_anchor_feature(model, im, box):
    anchor_region = me_extract_regions(im, box)
    anchor_region = process_regions(anchor_region)
    anchor_region = torch.Tensor(anchor_region)
    anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
    anchor_feature, _ = model(anchor_region)
    return anchor_feature


def Judge(model, anchor_feature, im, box):
    pos_region = me_extract_regions(np.array(im), box)
    pos_region = process_regions(pos_region)
    pos_region = torch.Tensor(pos_region)
    pos_region = (Variable(pos_region)).type(torch.FloatTensor).cuda()
    pos_feature, class_result = model(pos_region)

    class_result = torch.softmax(class_result, dim=1)
    threshold = 8.0828
    # threshold = 4.0
    ap_dist = torch.norm(anchor_feature - pos_feature, 2, dim=1).view(-1)
    if ap_dist < threshold:
        return True
    else:
        return False


######################################################################
model = ft_net(class_num=1400)
model.eval()
model = model.cuda()
model.load_state_dict(torch.load('metric_model/metric_model_21992.pt'))

anchor_feature = get_anchor_feature(model, anchor_im, anchor_box) #anchor_box: (1,4) x,y,w,h
result = Judge(model, anchor_feature, pos_im, pos_box)
print(result)

