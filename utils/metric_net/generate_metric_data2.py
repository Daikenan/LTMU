import os
import torch
from me_sample_generator import *
from PIL import Image
from metric_model import ft_net
from torch.autograd import Variable
import numpy as np
import cv2
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def get_anchor_feature(model, im, box):
    anchor_region = me_extract_regions(im, box)
    anchor_region = process_regions(anchor_region)
    anchor_region = torch.Tensor(anchor_region)
    anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
    anchor_feature, _ = model(anchor_region)
    # anchor_feature, _ = model(anchor_region)
    return anchor_feature

def process_regions(regions):
    # regions = np.squeeze(regions, axis=0)
    regions = regions / 255.0
    regions[:,:, :, 0] = (regions[:,:, :, 0] - 0.485) / 0.229
    regions[:,:, :, 1] = (regions[:,:, :, 1] - 0.456) / 0.224
    regions[:,:, :, 2] = (regions[:,:, :, 2] - 0.406) / 0.225
    regions = np.transpose(regions, (0,3, 1, 2))
    #regions = np.expand_dims(regions, axis=0)
    #regions = np.tile(regions, (2,1,1,1))
    return regions

def Judge( anchor_feature, pos_feature,flag):
    #threshold = 8.0828
    #threshold = 4.0
    pos_feature=pos_feature.repeat(anchor_feature.shape[0],1)
    ap_dist = torch.norm(anchor_feature - pos_feature, 2, dim=1).view(-1)
    del pos_feature
    if flag:#pos
        threshold = 8.0828
        #print('pos',ap_dist.min().cpu().detach().numpy(),ap_dist.max().cpu().detach().numpy())
        result=ap_dist<threshold
    else:#neg
        threshold = 4.0
        #print('neg', ap_dist.min().cpu().detach().numpy(), ap_dist.max().cpu().detach().numpy())
        result=ap_dist>threshold
    return result

def model_load(path):
    model = ft_net(class_num=1120)
    model.eval()
    model = model.cuda()
    model.load_state_dict(torch.load(path))
    tmp = np.random.rand(1, 3, 107, 107)
    tmp = (Variable(torch.Tensor(tmp))).type(torch.FloatTensor).cuda()
    model(tmp)
    return model

def judge_metric(model,anchor_box,im,target_feature,flag):#flag=1(pos),flag=0(neg)
    anchor_feature = get_anchor_feature(model, im, anchor_box)  # anchor_box: (1,4) x,y,w,h
    result = Judge(anchor_feature, target_feature,flag)
    result=result.cpu().numpy()
    result.dtype='bool'
    anchor_box=anchor_box[result]
    del anchor_feature
    return anchor_box

def get_target_feature(model,pos_box,im):
    pos_box = pos_box.reshape((1, 4))
    pos_region = me_extract_regions(im, pos_box)
    pos_region = process_regions(pos_region)
    pos_region = torch.Tensor(pos_region)
    pos_region = (Variable(pos_region)).type(torch.FloatTensor).cuda()
    pos_feature, _ = model(pos_region)#_ is class_result
    #class_result = torch.softmax(class_result, dim=1)
    return pos_feature

def coordinate_from_proportion(box, w ,h):
    coord_box = np.array([box[0]*w, box[1]*h, (box[2]-box[0])*w, (box[3]-box[1])*h])
    coord_box = coord_box.reshape((1, 4))
    return coord_box
# data_path = '/home/dkn/daikenan/dataset/VOT2019/lt2019/bike1/color/'
# img_list = os.listdir(data_path)
# img_list.sort()
# gt = np.loadtxt("/home/dkn/daikenan/dataset/VOT2019/lt2019/bike1/groundtruth.txt", delimiter=',')
#
# img_num = len(img_list)
# model = model_load('metric_model/metric_model_21992.pt')
# anchor_box = gt[0, :]
# anchor_box = anchor_box.reshape((1, 4))
# anchor_im = np.array(Image.open(data_path + img_list[0]))
# anchor_feature = get_anchor_feature(model, anchor_im, anchor_box) #anchor_box: (1,4) x,y,w,h
#
# for pos_index in range(img_num):
#     pos_box = gt[pos_index,:]
#     pos_box = pos_box.reshape((1,4))
#     pos_im = np.array(Image.open(data_path + img_list[pos_index]))
#
#
#     pos_feature = get_target_feature(model,pos_box, pos_im)
#     pos_feature = pos_feature.repeat(anchor_feature.shape[0], 1)
#     ap_dist = torch.norm(anchor_feature - pos_feature, 2, dim=1).view(-1)
#     print(ap_dist)


def eval_tracking(Dataset, video_spe=None, save=False, local_data=None, classes=None):
    if Dataset == "votlt":
        data_dir = '/home/daikenan/dataset/VOT18_long'
    elif Dataset == 'otb':
        data_dir = '/home/dkn/daikenan/dataset/OTB'
    elif Dataset == "votlt19":
        data_dir = '/home/dkn/data/lt2019'
    elif Dataset == "lasot":
        data_dir = os.path.join('/home/dkn/daikenan/dataset/LASOT/LaSOTBenchmark/', classes)
    sequence_list = os.listdir(data_dir)
    sequence_list.sort()
    sequence_list = [title for title in sequence_list if not title.endswith("txt")]
    if video_spe is not None:
        sequence_list = [video_spe]
    base_save_path = os.path.join('/media/dkn/Data2/dimp', local_data+'_add_metric', Dataset)
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    local_tracker_data_dir = '/media/dkn/Data2/dimp/dimp_ori'
    local_data_list = os.listdir(os.path.join(local_tracker_data_dir, Dataset))
    model = model_load('metric_model/metric_model_zj_57470.pt')

    for seq_id, video in enumerate(sequence_list):
        if Dataset == "votlt" or Dataset == "votlt19":
            sequence_dir = data_dir + '/' + video + '/color/'
            gt_dir = data_dir + '/' + video + '/groundtruth.txt'
        elif Dataset == "otb":
            sequence_dir = data_dir + '/' + video + '/img/'
            gt_dir = data_dir + '/' + video + '/groundtruth_rect.txt'
        elif Dataset == "lasot":
            sequence_dir = data_dir + '/' + video + '/img/'
            gt_dir = data_dir + '/' + video + '/groundtruth.txt'

        result_save_path = os.path.join(base_save_path, video+'.txt')
        if os.path.exists(result_save_path) or (video+'.txt' not in local_data_list):
            continue
        image_list = os.listdir(sequence_dir)
        image_list.sort()
        image_list = [im for im in image_list if im.endswith("jpg") or im.endswith("jpeg")]
        try:
            groundtruth = np.loadtxt(gt_dir, delimiter=',')
        except:
            groundtruth = np.loadtxt(gt_dir)
        image_dir = sequence_dir + image_list[0]
        image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]
        num_frames = len(image_list)
        dis_all = np.zeros((num_frames, 1))
        local_txt_dir = os.path.join(local_tracker_data_dir, Dataset, video + '.txt')
        local_txt = np.loadtxt(local_txt_dir, delimiter=',')
        anchor_box = coordinate_from_proportion(local_txt[0, 0:4], w, h)
        anchor_feature = get_anchor_feature(model, image, anchor_box)
        local_map_dir = os.path.join(local_tracker_data_dir, Dataset, video + '_map.npy')
        local_map = np.load(local_map_dir)
        score_all = np.max(np.max(np.array(local_map), axis=1), axis=1)
        score_all = np.expand_dims(score_all,axis=1)



        for im_id in range(1, len(image_list)):
            imagefile = sequence_dir + image_list[im_id]
            image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)
            pos_box = coordinate_from_proportion(local_txt[im_id, 0:4], w, h)
            try:
                instance_feature = get_target_feature(model, pos_box, image)
                ap_dist = torch.norm(anchor_feature - instance_feature, 2, dim=1).view(-1)
            except:
                ap_dist = torch.norm(anchor_feature, 2, dim=1).view(-1)
            print(ap_dist.data.cpu().numpy())
            dis_all[im_id] = ap_dist.data.cpu().numpy()[0]
            print("%d: " % seq_id + video + ": %d /" % im_id + "%d" % len(image_list))
        bBoxes = np.concatenate([local_txt, score_all, dis_all], axis=1)

        if save:
            np.savetxt(result_save_path, bBoxes, fmt="%.8f,%.8f,%.8f,%.8f,%d,%.8f,%.8f,%.8f")
        # np.savetxt(os.path.join('/home/daikenan/Desktop/MDNet_tf/', video+'.txt'), bBoxes2, fmt="%.6f,%.6f,%.6f,%.6f")



def main(_):
    lasot_dir = '/home/dkn/daikenan/dataset/LASOT/LaSOTBenchmark/'
    classes = os.listdir(lasot_dir)
    classes.sort()
    # for i in range(len(videos)):
    #     eval_tracking('votlt19', p=p, video_spe=videos[i])
    # p = p_config()
    # p.name = 'test1'
    for c in classes:
        eval_tracking('lasot', classes=c, save=True, local_data='dimp_ori')
    # eval_tracking('votlt19', local_data='vot_tracker_eval_data', save=True)


if __name__ == '__main__':
    tf.app.run()
