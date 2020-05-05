import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


p = p_config()
p.tracker='Dimp'
p.model_dir = 'dimp_mu_1'
p.name = p.tracker + '_' + p.model_dir
eval_tracking('votlt19', p=p, mode='test')

