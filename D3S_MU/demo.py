import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test D3S+MU
p = p_config()
p.tracker = 'D3S_MU'
p.save_results = False
eval_tracking('demo', p=p, mode='test')

# # test D3S
# p = p_config()
# p.tracker = 'D3S'
# p.save_results = False
# p.name = p.tracker
# eval_tracking('demo', p=p, mode='test')
