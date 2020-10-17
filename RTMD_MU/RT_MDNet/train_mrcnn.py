import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0,'./modules')
from data_prov import *
from model import *
from pretrain_options import *
from tracker import *
import numpy as np

import argparse


def set_optimizer(model, lr_base, lr_mult=pretrain_opts['lr_mult'], momentum=pretrain_opts['momentum'], w_decay=pretrain_opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)

    if set_type == 'OTB':
        img_list = sorted([seq_path + '/img/' + p for p in os.listdir(seq_path + '/img') if os.path.splitext(p)[1] == '.jpg'])

        if (seqname == 'Jogging') or (seqname == 'Skating2'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect.1.txt')
        elif seqname =='Human4':
            gt = np.loadtxt(seq_path + '/groundtruth_rect.2.txt', delimiter=',')
        elif (seqname == 'BlurBody')  or (seqname == 'BlurCar1') or (seqname == 'BlurCar2') or (seqname == 'BlurCar3') \
                or (seqname == 'BlurCar4') or (seqname == 'BlurFace') or (seqname == 'BlurOwl') or (seqname == 'Board') \
                or (seqname == 'Box')   or (seqname == 'Car4')  or (seqname == 'CarScale') or (seqname == 'ClifBar') \
                or (seqname == 'Couple')  or (seqname == 'Crossing')  or (seqname == 'Dog') or (seqname == 'FaceOcc1') \
                or (seqname == 'Girl') or (seqname == 'Rubik') or (seqname == 'Singer1') or (seqname == 'Subway') \
                or (seqname == 'Surfer') or (seqname == 'Sylvester') or (seqname == 'Toy') or (seqname == 'Twinnings') \
                or (seqname == 'Vase') or (seqname == 'Walking') or (seqname == 'Walking2') or (seqname == 'Woman')   :
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt')
        elif (seqname == 'Freeman4') or (seqname == 'Diving') or (seqname =='Freeman3') or (seqname =='Football1'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect_revise.txt', delimiter=',')
        else:
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

        if seqname == 'David':
            img_list = img_list[300:]
            # gt = gt[300:,:]
        if seqname == 'Football1':
            img_list = img_list[0:73]
        if seqname == 'Freeman3':
            img_list = img_list[0:459]
        if seqname == 'Freeman4':
            img_list = img_list[0:282]

    elif set_type=='VOT/2016':
        img_list = sorted([seq_path + '/'+p for p in os.listdir(seq_path) if os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')

        ##polygon to rect
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    return img_list, gt


def train_mdnet():

    ## set image directory
    if pretrain_opts['set_type'] == 'OTB':
        img_home = '/home/ilchae/dataset/tracking/OTB/'
        data_path = './otb-vot15.pkl'
    if pretrain_opts['set_type'] == 'VOT':
        img_home = '/home/ilchae/dataset/tracking/VOT/'
        data_path = './vot-otb.pkl'
    if pretrain_opts['set_type'] == 'IMAGENET':
        img_home = '/home/ilchae/dataset/ILSVRC/Data/VID/train/'
        data_path = './modules/imagenet_refine.pkl'

    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)


    K = len(data)

    ## Init model ##
    model = MDNet(pretrain_opts['init_model_path'], K)
    if pretrain_opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)

    if pretrain_opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(pretrain_opts['ft_layers'])
    model.train()

    dataset = [None] * K
    for k, (seqname, seq) in enumerate(data.iteritems()):
        img_list = seq['images']
        gt = seq['gt']
        if pretrain_opts['set_type'] == 'OTB':
            img_dir = os.path.join(img_home, seqname+'/img')
        if pretrain_opts['set_type'] == 'VOT':
            img_dir = img_home + seqname
        if pretrain_opts['set_type'] == 'IMAGENET':
            img_dir = img_home + seqname
        dataset[k]=RegionDataset(img_dir,img_list,gt,model.receptive_field,pretrain_opts)


    ## Init criterion and optimizer ##
    binaryCriterion = BinaryLoss()
    interDomainCriterion = nn.CrossEntropyLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, pretrain_opts['lr'])

    best_score = 0.
    batch_cur_idx = 0
    for i in range(pretrain_opts['n_cycles']):
        print "==== Start Cycle %d ====" % (i)
        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        totalTripleLoss = np.zeros(K)
        totalInterClassLoss = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            try:
                cropped_scenes, pos_rois, neg_rois= dataset[k].next()
            except:
                continue

            try:
                for sidx in range(0, len(cropped_scenes)):
                    cur_scene = cropped_scenes[sidx]
                    cur_pos_rois = pos_rois[sidx]
                    cur_neg_rois = neg_rois[sidx]

                    cur_scene = Variable(cur_scene)
                    cur_pos_rois = Variable(cur_pos_rois)
                    cur_neg_rois = Variable(cur_neg_rois)
                    if pretrain_opts['use_gpu']:
                        cur_scene = cur_scene.cuda()
                        cur_pos_rois = cur_pos_rois.cuda()
                        cur_neg_rois = cur_neg_rois.cuda()
                    cur_feat_map = model(cur_scene, k, out_layer='conv3')

                    cur_pos_feats = model.roi_align_model(cur_feat_map, cur_pos_rois)
                    cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                    cur_neg_feats = model.roi_align_model(cur_feat_map, cur_neg_rois)
                    cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)

                    if sidx == 0:
                        pos_feats = [cur_pos_feats]
                        neg_feats = [cur_neg_feats]
                    else:
                        pos_feats.append(cur_pos_feats)
                        neg_feats.append(cur_neg_feats)
                feat_dim = cur_neg_feats.size(1)
                pos_feats = torch.stack(pos_feats,dim=0).view(-1,feat_dim)
                neg_feats = torch.stack(neg_feats,dim=0).view(-1,feat_dim)
            except:
                continue


            pos_score = model(pos_feats, k, in_layer='fc4')
            neg_score = model(neg_feats, k, in_layer='fc4')

            cls_loss = binaryCriterion(pos_score, neg_score)

            ## inter frame classification

            interclass_label = Variable(torch.zeros((pos_score.size(0))).long())
            if opts['use_gpu']:
                interclass_label = interclass_label.cuda()
            total_interclass_score = pos_score[:,1].contiguous()
            total_interclass_score = total_interclass_score.view((pos_score.size(0),1))

            K_perm = np.random.permutation(K)
            K_perm = K_perm[0:100]
            for cidx in K_perm:
                if k == cidx:
                    continue
                else:
                    interclass_score = model(pos_feats, cidx, in_layer='fc4')
                    total_interclass_score = torch.cat((total_interclass_score,interclass_score[:,1].contiguous().view((interclass_score.size(0),1))),dim=1)

            interclass_loss = interDomainCriterion(total_interclass_score, interclass_label)
            totalInterClassLoss[k] = interclass_loss.data[0]

            (cls_loss+0.1*interclass_loss).backward()

            batch_cur_idx+=1
            if (batch_cur_idx%pretrain_opts['seqbatch_size'])==0:
                torch.nn.utils.clip_grad_norm(model.parameters(), pretrain_opts['grad_clip'])
                optimizer.step()
                model.zero_grad()
                batch_cur_idx = 0

            ## evaulator
            prec[k] = evaluator(pos_score, neg_score)
            ## computation latency
            toc = time.time() - tic

            print "Cycle %2d, K %2d (%2d), BinLoss %.3f, Prec %.3f, interLoss %.3f, Time %.3f" % \
                      (i, j, k, cls_loss.data[0], prec[k], totalInterClassLoss[k], toc)

        cur_score = prec.mean()
        try:
            total_miou = sum(total_iou)/len(total_iou)
        except:
            total_miou = 0.
        print "Mean Precision: %.3f Triple Loss: %.3f Inter Loss: %.3f IoU: %.3f" % (prec.mean(), cur_triple_loss, totalInterClassLoss.mean(),total_miou)
        if cur_score > best_score:
            best_score = cur_score
            if pretrain_opts['use_gpu']:
                model = model.cpu()
            states = {'shared_layers': model.layers.state_dict()}
            print "Save model to %s" % pretrain_opts['model_path']
            torch.save(states, pretrain_opts['model_path'])
            if pretrain_opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'VOT' )
    parser.add_argument("-padding_ratio", default = 5., type =float)
    parser.add_argument("-model_path", default =".models/rt_mdnet.pth", help = "model path")
    parser.add_argument("-frame_interval", default = 1, type=int, help="frame interval in batch. ex) interval=1 -> [1 2 3 4 5], interval=2 ->[1 3 5]")
    parser.add_argument("-init_model_path", default="./models/imagenet-vgg-m.mat")
    parser.add_argument("-batch_frames", default = 8, type = int)
    parser.add_argument("-lr", default=0.0001, type = float)
    parser.add_argument("-batch_pos",default = 64, type = int)
    parser.add_argument("-batch_neg", default = 196, type = int)
    parser.add_argument("-n_cycles", default = 1000, type = int )
    parser.add_argument("-adaptive_align", default = True, action = 'store_false')
    parser.add_argument("-seqbatch_size", default=50, type=int)

    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ##option setting
    pretrain_opts['set_type'] = args.set_type
    pretrain_opts['padding_ratio']=args.padding_ratio
    pretrain_opts['padded_img_size']=pretrain_opts['img_size']*int(pretrain_opts['padding_ratio'])
    pretrain_opts['model_path']=args.model_path
    pretrain_opts['frame_interval'] = args.frame_interval
    pretrain_opts['init_model_path'] = args.init_model_path
    pretrain_opts['batch_frames'] = args.batch_frames
    pretrain_opts['lr'] = args.lr
    pretrain_opts['batch_pos'] = args.batch_pos  # original = 64
    pretrain_opts['batch_neg'] = args.batch_neg  # original = 192
    pretrain_opts['n_cycles'] = args.n_cycles
    pretrain_opts['adaptive_align']=args.adaptive_align
    pretrain_opts['seqbatch_size'] = args.seqbatch_size
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################

    print pretrain_opts
    train_mdnet()

