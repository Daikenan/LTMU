import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test DiMPMU
p = p_config()
p.tracker = 'Dimp_MU'
p.save_results = False
p.name = p.tracker + '_' + p.model_dir
eval_tracking('demo', p=p, mode='test')

# # test DiMP
# p = p_config()
# p.tracker = 'Dimp'
# p.save_results = False
# p.name = p.tracker
# eval_tracking('demo', p=p, mode='test')