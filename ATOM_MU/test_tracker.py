import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test ATOM+MU
p = p_config()
p.save_training_data = False
p.tracker = 'ATOM_MU'
p.checkpoint = 20000
p.name = p.tracker

# # test ATOM
# p = p_config()
# p.save_training_data = False
# p.tracker = 'ATOM'
# p.name = p.tracker

eval_tracking('tlp', p=p, mode='all')
eval_tracking('lasot', p=p, mode='test')
eval_tracking('votlt19', p=p, mode='all')

