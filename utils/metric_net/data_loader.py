import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from sample_generator import *
from PIL import Image
import torch.utils.data as torch_dataset
from model import ft_net
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.autograd import Variable
from torchvision import transforms
from tensorboardX import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin = margin, p = 2)
        self.class_loss = nn.CrossEntropyLoss()
    def forward(self, anchor, pos, neg, hard_neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor - pos, 2, dim = 1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim = 1).view(-1)
            ahn_dist = torch.norm(anchor - hard_neg, 2, dim = 1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y) / 2.0 + self.Loss(ahn_dist - ap_dist, y) / 2.0

        else:
            loss = self.Loss(anchor, pos, neg)

        return loss

class MixedLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = None):
        super(MixedLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin = margin, p = 2)
        self.class_loss = nn.CrossEntropyLoss()
    def forward(self, anchor, pos, neg, hard_neg, an_class, p_class, neg_class, hard_neg_class, class_ids, hard_class_ids):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor - pos, 2, dim = 1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim = 1).view(-1)
            ahn_dist = torch.norm(anchor - hard_neg, 2, dim = 1).view(-1)
            matching_loss = self.Loss(an_dist - ap_dist, y) / 2.0 + self.Loss(ahn_dist - ap_dist, y) / 2.0

            class_loss1 = self.class_loss(an_class, class_ids)
            class_loss2 = self.class_loss(p_class, class_ids)
            class_loss3 = self.class_loss(hard_neg_class, hard_class_ids)
            #class_loss = (class_loss1+class_loss2 + class_loss3)/3.0

            neg_class = torch.softmax(neg_class, dim=1)
            neg_class_loss = torch.sum(-1.0/1400 * torch.log(neg_class),dim=1)
            neg_class_loss = torch.mean(neg_class_loss, dim=0)

            class_loss = (class_loss1+ class_loss2 + class_loss3 + neg_class_loss) / 4.0

            loss = matching_loss + class_loss

        else:
            loss = self.Loss(anchor, pos, neg)

        return loss, matching_loss, class_loss

class ValidationDataset(torch_dataset.Dataset):
    def __init__(self, src_path):
        #src_path = '/home/xiaobai/Documents/LaSOT/LaSOTBenchmark/'
        self.data_dict = dict()
        seq_list = os.listdir(src_path)
        seq_list.sort()
        seq_lists = []
        #class_id = 0
        for seq_id, seq_name in enumerate(seq_list):
            self.data_dict[seq_name] = dict()
            self.data_dict[seq_name]['pos'] = os.listdir(src_path + seq_name + '/pos/')
            self.data_dict[seq_name]['neg'] = os.listdir(src_path + seq_name + '/neg/')
            self.data_dict[seq_name]['pos_num'] = len(self.data_dict[seq_name]['pos'])
            self.data_dict[seq_name]['neg_num'] = len(self.data_dict[seq_name]['neg'])
            #class_id += 1
            # print 'Loading sequences: ', seq_id, '/', 50

        self.src_path = src_path
        self.keys = self.data_dict.keys()
        self.seq_num = len(self.data_dict.keys())
        transform_train_list = [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.data_transforms = transforms.Compose(transform_train_list)

    def process_regions(self, regions):
        #regions = np.squeeze(regions, axis=0)
        regions = regions / 255.0
        regions[:,:,0] = (regions[:,:,0] - 0.485) / 0.229
        regions[:,:,1] = (regions[:,:,1] - 0.456) / 0.224
        regions[:,:,2] = (regions[:,:,2] - 0.406) / 0.225
        regions = np.transpose(regions, (2,0,1))
        return regions

    def __getitem__(self, index):
        #seq_id = np.random.randint(low=0, high = self.seq_num-1, size=[1])[0]
        seq_name = self.keys[index]

        index1 = np.random.randint(low=0,high=self.data_dict[seq_name]['pos_num'],size=[1,])[0]
        anchor_regions = np.array(Image.open(self.src_path + seq_name + '/pos/' + self.data_dict[seq_name]['pos'][index1]))
        anchor_regions = self.process_regions(anchor_regions)

        index1 = np.random.randint(low=0,high=self.data_dict[seq_name]['pos_num'],size=[1,])[0]
        pos_regions = np.array(Image.open(self.src_path + seq_name + '/pos/' + self.data_dict[seq_name]['pos'][index1]))
        pos_regions = self.process_regions(pos_regions)

        if self.data_dict[seq_name]['neg_num'] > 0:
            index1 = np.random.randint(low=0,high=self.data_dict[seq_name]['neg_num'],size=[1,])[0]
            neg_regions = np.array(Image.open(self.src_path + seq_name + '/neg/' + self.data_dict[seq_name]['neg'][index1]))
            neg_regions = self.process_regions(neg_regions)
        else:
            index2 = np.random.randint(low=0, high=self.seq_num, size=[1])[0]
            while index2 == index:
                index2 = np.random.randint(low=0, high=self.seq_num, size=[1])[0]
            seq_name = self.keys[index2]
            index1 = np.random.randint(low=0, high=self.data_dict[seq_name]['pos_num'], size=[1, ])[0]
            neg_regions = np.array(Image.open(self.src_path + seq_name + '/pos/' + self.data_dict[seq_name]['pos'][index1]))
            neg_regions = self.process_regions(neg_regions)

        index2 = np.random.randint(low=0,high=self.seq_num, size = [1])[0]
        while index2 == index:
            index2 = np.random.randint(low=0, high=self.seq_num, size=[1])[0]
        seq_name = self.keys[index2]
        index1 = np.random.randint(low=0,high=self.data_dict[seq_name]['pos_num'],size=[1,])[0]
        hard_neg_regions = np.array(Image.open(self.src_path + seq_name + '/pos/' + self.data_dict[seq_name]['pos'][index1]))
        hard_neg_regions = self.process_regions(hard_neg_regions)

        return anchor_regions, pos_regions, neg_regions, hard_neg_regions

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.seq_num

class CustomDataset(torch_dataset.Dataset):
    def __init__(self, src_path):
        #src_path = '/home/xiaobai/Documents/LaSOT/LaSOTBenchmark/'
        self.data_dict = dict()
        folder_list = os.listdir(src_path)
        folder_list.sort()
        seq_lists = []
        class_id = 0
        for folder_id, folder in enumerate(folder_list):
            #seq_list = sorted([src_path + folder + '/' + seq for seq in seq_list])
            seq_list = os.listdir(src_path + folder)
            seq_list = sorted([folder + '/' + seq for seq in seq_list])
            for seq_id, seq_name in enumerate(seq_list):
                self.data_dict[seq_name] = dict()
                self.data_dict[seq_name]['pos'] = os.listdir(src_path + seq_name + '/pos/')
                self.data_dict[seq_name]['neg'] = os.listdir(src_path + seq_name + '/neg/')
                self.data_dict[seq_name]['pos_num'] = len(self.data_dict[seq_name]['pos'])
                self.data_dict[seq_name]['neg_num'] = len(self.data_dict[seq_name]['neg'])
                self.data_dict[seq_name]['class_id'] = class_id
                class_id += 1
                # print 'Loading sequences: ', class_id, '/', 1400

        self.src_path = src_path
        self.keys = self.data_dict.keys()
        self.seq_num = len(self.data_dict.keys())
        transform_train_list = [
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        self.data_transforms = transforms.Compose(transform_train_list)

    def process_regions(self, regions):
        #regions = np.squeeze(regions, axis=0)
        regions = regions / 255.0
        regions[:,:,0] = (regions[:,:,0] - 0.485) / 0.229
        regions[:,:,1] = (regions[:,:,1] - 0.456) / 0.224
        regions[:,:,2] = (regions[:,:,2] - 0.406) / 0.225
        regions = np.transpose(regions, (2,0,1))
        return regions

    def __getitem__(self, index):
        #seq_id = np.random.randint(low=0, high = self.seq_num-1, size=[1])[0]
        seq_name = self.keys[index]

        class_id = self.data_dict[seq_name]['class_id']
        class_id = np.array([class_id,])

        index1 = np.random.randint(low=0,high=self.data_dict[seq_name]['pos_num'],size=[1,])[0]
        anchor_regions = np.array(Image.open(self.src_path + seq_name + '/pos/' + self.data_dict[seq_name]['pos'][index1]))
        anchor_regions = self.process_regions(anchor_regions)

        index1 = np.random.randint(low=0,high=self.data_dict[seq_name]['pos_num'],size=[1,])[0]
        pos_regions = np.array(Image.open(self.src_path + seq_name + '/pos/' + self.data_dict[seq_name]['pos'][index1]))
        pos_regions = self.process_regions(pos_regions)

        if self.data_dict[seq_name]['neg_num'] > 0:
            index1 = np.random.randint(low=0,high=self.data_dict[seq_name]['neg_num'],size=[1,])[0]
            neg_regions = np.array(Image.open(self.src_path + seq_name + '/neg/' + self.data_dict[seq_name]['neg'][index1]))
            neg_regions = self.process_regions(neg_regions)
        else:
            index2 = np.random.randint(low=0, high=self.seq_num, size=[1])[0]
            while index2 == index:
                index2 = np.random.randint(low=0, high=self.seq_num, size=[1])[0]
            seq_name = self.keys[index2]
            index1 = np.random.randint(low=0, high=self.data_dict[seq_name]['pos_num'], size=[1, ])[0]
            neg_regions = np.array(Image.open(self.src_path + seq_name + '/pos/' + self.data_dict[seq_name]['pos'][index1]))
            neg_regions = self.process_regions(neg_regions)

        index2 = np.random.randint(low=0,high=self.seq_num, size = [1])[0]
        while index2 == index:
            index2 = np.random.randint(low=0, high=self.seq_num, size=[1])[0]

        seq_name = self.keys[index2]
        index1 = np.random.randint(low=0,high=self.data_dict[seq_name]['pos_num'],size=[1,])[0]
        hard_neg_regions = np.array(Image.open(self.src_path + seq_name + '/pos/' + self.data_dict[seq_name]['pos'][index1]))
        hard_neg_regions = self.process_regions(hard_neg_regions)

        hard_class_id = self.data_dict[seq_name]['class_id']
        hard_class_id = np.array([hard_class_id,])

        return anchor_regions, pos_regions, neg_regions, hard_neg_regions, class_id, hard_class_id

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.seq_num

LR_Rate = 1e-3
model = ft_net(class_num=1400)
ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1 * LR_Rate},
    {'params': model.classifier.parameters(), 'lr': LR_Rate}
], weight_decay=5e-4, momentum=0.9, nesterov=True)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
#criterion = nn.CrossEntropyLoss()
criterion = MixedLoss()
validation_criterion = TripletLoss()
model = model.cuda()

model.load_state_dict(torch.load('metric_model/metric_model_19448.pt'))

writer_path = './summary'
if not os.path.exists(writer_path):
    os.mkdir(writer_path)
writer = SummaryWriter(writer_path)

BatchSize = 16
M = 2
im_per_seq = 4
data_path = '/home/xiaobai/Documents/LaSOT_crops/'
validation_data_path = '/home/xiaobai/Documents/lt2019_crops/'
dataset = CustomDataset(data_path)
validation_dataset = ValidationDataset(validation_data_path)
train_loader = torch_dataset.DataLoader(dataset=dataset, batch_size = BatchSize, shuffle=True)
validation_loader = torch_dataset.DataLoader(dataset=validation_dataset, batch_size = BatchSize, shuffle=True)
save_path = './metric_model'
if not os.path.exists(save_path):
    os.mkdir(save_path)
#time2 = time.time()

#15K iteration lr->1e-3 19448 lr->1e-4
iter = 20144
for epoch in range(1000):
    # if iter > 15000*M:
    #     LR_Rate = 1e-3
    #     optimizer_ft = optim.SGD([
    #         {'params': base_params, 'lr': 0.1 * LR_Rate},
    #         {'params': model.classifier.parameters(), 'lr': LR_Rate}
    #     ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    # if iter > 19400*M:
    #     LR_Rate = 1e-4
    #     optimizer_ft = optim.SGD([
    #         {'params': base_params, 'lr': 0.1 * LR_Rate},
    #         {'params': model.classifier.parameters(), 'lr': LR_Rate}
    #     ], weight_decay=5e-4, momentum=0.9, nesterov=True)


    validation_loss = 0
    validation_iter = 0
    for anchor_regions, pos_regions, neg_regions, hard_neg_regions in validation_loader:
        # time1 = time.time()
        # print "read data time: ", time1 - time2
        pos_regions = (Variable(pos_regions)).type(torch.FloatTensor).cuda()
        anchor_regions = (Variable(anchor_regions)).type(torch.FloatTensor).cuda()
        neg_regions = (Variable(neg_regions)).type(torch.FloatTensor).cuda()
        hard_neg_regions = (Variable(hard_neg_regions)).type(torch.FloatTensor).cuda()

        optimizer_ft.zero_grad()
        anchor_metric, anchor_class = model(anchor_regions)
        pos_metric, pos_class = model(pos_regions)
        neg_metric, neg_class = model(neg_regions)
        hard_neg_metric, hard_neg_class = model(hard_neg_regions)
        loss = validation_criterion(anchor_metric, pos_metric, neg_metric, hard_neg_metric)
        validation_loss = (validation_loss * validation_iter + loss.item()) / (validation_iter + 1)
        validation_iter += 1

    writer.add_scalar('validation_loss', validation_loss, epoch)

    # print "epoch: ", epoch, ", iteration: ", iter, ", validation loss: ", validation_loss

    for anchor_regions, pos_regions, neg_regions, hard_neg_regions, class_ids, hard_class_ids in train_loader:
        #time1 = time.time()
        #print "read data time: ", time1 - time2
        pos_regions = (Variable(pos_regions)).type(torch.FloatTensor).cuda()
        anchor_regions = (Variable(anchor_regions)).type(torch.FloatTensor).cuda()
        neg_regions = (Variable(neg_regions)).type(torch.FloatTensor).cuda()
        hard_neg_regions = (Variable(hard_neg_regions)).type(torch.FloatTensor).cuda()
        class_ids = Variable(class_ids.cuda())
        hard_class_ids = Variable(hard_class_ids.cuda())

        optimizer_ft.zero_grad()
        anchor_metric,anchor_class = model(anchor_regions)
        pos_metric,pos_class = model(pos_regions)
        neg_metric,neg_class = model(neg_regions)
        hard_neg_metric, hard_neg_class = model(hard_neg_regions)
        class_ids = torch.squeeze(class_ids, 1)
        hard_class_ids = torch.squeeze(hard_class_ids, 1)
        loss, matching_loss, class_loss = criterion(anchor_metric, pos_metric, neg_metric, hard_neg_metric, anchor_class, pos_class, neg_class, hard_neg_class, class_ids, hard_class_ids)
        loss.backward()
        writer.add_scalar('loss', loss.cpu(), iter)
        writer.add_scalar('matching loss', matching_loss.cpu(), iter)
        writer.add_scalar('classification loss', class_loss.cpu(), iter)

        iter += 1
        optimizer_ft.step()
        # print "epoch: ", epoch, ", iteration: ", iter, ", loss: ", loss.item(), ", matching loss: ", matching_loss.item(), ", classification loss: ", class_loss.item()
        #time2 = time.time()
        #print "train time:", time2 - time1
    if np.mod(epoch, 20) == 0:
        torch.save(model.state_dict(), save_path+ '/' + save_path+'_'+str(iter)+'.pt')

writer.close()