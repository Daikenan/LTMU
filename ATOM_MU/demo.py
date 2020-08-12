import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test ATOM+MU
p = p_config()
p.tracker = 'ATOM_MU'
p.name = p.tracker
eval_tracking('demo', p=p, mode='test')

# # test ATOM
# p = p_config()
# p.tracker = 'ATOM'
# p.save_results = False
# p.name = p.tracker
# eval_tracking('demo', p=p, mode='test')
