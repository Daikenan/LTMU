import os
from run_tracker import eval_tracking, p_config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


p = p_config()
p.model_dir = 'test3_new'
p.checkpoint = 14000
p.start_frame = 20
p.name = 'test3_2_' + str(p.checkpoint) + '_' + str(p.start_frame)
eval_tracking('votlt19', p=p,)



p = p_config()
p.model_dir = 'test3_new'
p.checkpoint = 220000
p.start_frame = 200
p.name = 'test3_2_' + str(p.checkpoint) + '_' + str(p.start_frame)
eval_tracking('votlt19', p=p,)