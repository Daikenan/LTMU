import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test RTMD+MU
p = p_config()
p.tracker = 'RTMD_MU'
p.save_results = True
p.name = p.tracker
eval_tracking('lasot', p=p, mode='test')
p.save_training_data = False
eval_tracking('tlp', p=p, mode='all')
eval_tracking('votlt19', p=p, mode='all')

# # test RTMD
# p = p_config()
# p.tracker = 'RTMD'
# p.name = p.tracker
# eval_tracking('lasot', p=p, mode='test')
# p.save_training_data = False
# eval_tracking('tlp', p=p, mode='all')
# eval_tracking('votlt19', p=p, mode='all')
