import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# test PrDiMP+MU
p = p_config()
p.tracker = 'PrDimp_MU'
p.name = p.tracker
eval_tracking('demo', p=p)

# # test PrDiMP
# p = p_config()
# p.tracker = 'PrDimp'
# p.name = p.tracker
# eval_tracking('demo', p=p)
