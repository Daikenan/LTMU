import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test SuperDiMP+MU
p = p_config()
p.tracker = 'Super_Dimp_MU'
p.save_results = False
p.name = p.tracker
eval_tracking('demo', p=p)

# # test SuperDiMP
# p = p_config()
# p.tracker = 'Super_Dimp'
# p.save_results = False
# p.name = p.tracker
# eval_tracking('demo', p=p)
