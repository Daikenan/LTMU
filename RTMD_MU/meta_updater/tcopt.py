from collections import OrderedDict

tcopts = OrderedDict()

tcopts['GPU_num'] = 2
tcopts['GPU_ID'] = [1]

tcopts['win_size'] = 107
tcopts['img_size'] = 600
tcopts['win_padding_scale'] = 2

# training batch
tcopts['batch_size'] = 64  # peer gpu
tcopts['num_examples_per_epoch'] = 1400000
tcopts['NUM_EPOCHS_PER_DECAY'] = 2.5
tcopts['num_of_step'] = 400000
tcopts['keep_checkpoint_every_n_hours'] = 1
tcopts['initial_lr'] = 1e-4
tcopts['lr_decay_factor'] = 0.1
tcopts['num_threads'] = 8
tcopts['capacity'] = 32


tcopts['train_dir'] = './save_path'
tcopts['model_name'] = 'model.ckpt'

# slim config
tcopts['save_summaries_secs'] = 120
tcopts['save_interval_secs'] = 600
# NOT use slim config

tcopts['summary_save_interval'] = 1000
tcopts['model_save_interval'] = 100000
tcopts['eval_interval'] = 1000

# lstm
tcopts['lstm_train_dir'] = './rtmd_MU_0_5_3'
tcopts['pos_thr'] = 0.5
tcopts['lstm_initial_lr'] = 1e-4
tcopts['lstm_decay_steps'] = 100000
tcopts['lstm_input'] = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11]
tcopts['sampling_interval'] = 4
tcopts['lstm_num_input'] = len(tcopts['lstm_input'])
tcopts['lstm_num_classes'] = 2
tcopts['time_steps'] = 20
tcopts['start_frame'] = 20
tcopts['save_pos_thr'] = 0.90
tcopts['save_neg_thr'] = 0.90