import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test PrDiMP+MU
p = p_config()
p.tracker = 'PrDimp_MU'
p.model_dir = 'prdimp_mu_1'
p.name = p.tracker + '_' + p.model_dir

eval_tracking('lasot', p=p, mode='test')
eval_tracking('tlp', p=p, mode='all')
eval_tracking('votlt19', p=p, mode='all')

# # test PrDiMP
# p = p_config()
# p.tracker = 'PrDimp'
# p.name = p.tracker
# eval_tracking('lasot', p=p, mode='test')
# eval_tracking('tlp', p=p, mode='all')
# eval_tracking('votlt19', p=p, mode='all')
