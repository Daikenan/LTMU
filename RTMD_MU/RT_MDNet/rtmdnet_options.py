from collections import OrderedDict
from local_path import base_path
rt_opts = OrderedDict()
rt_opts['use_gpu'] = True


rt_opts['model_path'] = './RT_MDNet/models/rt-mdnet.pth'

rt_opts['img_size'] = 107
rt_opts['padding'] = 1.2
rt_opts['jitter'] = True
rt_opts['result_path']= base_path + 'RT_MDNet/result.npy'
rt_opts['adaptive_align']=True
rt_opts['batch_pos'] = 32
rt_opts['batch_neg'] = 96
rt_opts['batch_neg_cand'] = 1024
rt_opts['batch_test'] = 256

rt_opts['n_samples'] = 256
rt_opts['trans_f'] = 0.6
rt_opts['scale_f'] = 1.05
rt_opts['trans_f_expand'] = 1.4

rt_opts['n_bbreg'] = 1000
rt_opts['overlap_bbreg'] = [0.6, 1]
rt_opts['scale_bbreg'] = [1, 2]

rt_opts['lr_init'] = 0.0001 # original = 0.0001
rt_opts['maxiter_init'] = 50 # original = 30
rt_opts['n_pos_init'] = 500
rt_opts['n_neg_init'] = 5000
rt_opts['overlap_pos_init'] = [0.7, 1]
rt_opts['overlap_neg_init'] = [0, 0.5]

rt_opts['lr_update'] = 0.0003 # original = 0.0002
rt_opts['maxiter_update'] = 15 # original = 15
rt_opts['n_pos_update'] = 50 # original = 50
rt_opts['n_neg_update'] = 200
rt_opts['overlap_pos_update'] = [0.7, 1]
rt_opts['overlap_neg_update'] = [0, 0.3]

rt_opts['success_thr'] = 0. # original = 0
rt_opts['n_frames_short'] = 20
rt_opts['n_frames_long'] = 150
rt_opts['long_interval'] = 10

rt_opts['w_decay'] = 0.0005 # original = 0.0005
rt_opts['momentum'] = 0.9
rt_opts['grad_clip'] = 10 # original = 10
rt_opts['lr_mult'] = {'fc6':10}
rt_opts['ft_layers'] = ['fc']

rt_opts['visual_log'] = False



