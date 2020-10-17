import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test RTMD+MU
p = p_config()
p.tracker = 'RTMD_MU'
p.save_results = False
p.name = p.tracker
eval_tracking('demo', p=p)


