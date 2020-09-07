import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test DiMPMU
p = p_config()
p.tracker = 'Super_Dimp_MU'
p.model_dir = 'super_dimp_mu_1'
p.save_training_data = True
p.name = p.tracker + '_' + p.model_dir
eval_tracking('lasot', p=p, mode='test')
p.save_training_data = False
eval_tracking('tlp', p=p, mode='all')
eval_tracking('votlt19', p=p, mode='all')

# # test DiMP
# p = p_config()
# p.tracker = 'Dimp'
# p.name = p.tracker
# eval_tracking('votlt19', p=p, mode='test')