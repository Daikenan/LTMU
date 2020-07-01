import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test DiMP_LTMU
p = p_config()
p.save_results = True
p.name = 'DiMP_LTMU'
eval_tracking('votlt19', p=p)
