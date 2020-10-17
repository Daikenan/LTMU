from collections import OrderedDict

tcopts = OrderedDict()

# training batch
tcopts['batch_size'] = 32  # peer gpu
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

tcopts['summary_save_interval'] = 50
tcopts['model_save_interval'] = 5000
tcopts['eval_interval'] = 500

# lstm
tcopts['lstm_train_dir'] = './prdimp_mu_1'
tcopts['train_data_dir'] = '../results/D3S/lasot/train_data'
tcopts['pos_thr'] = 0.5
tcopts['lstm_initial_lr'] = 1e-3
tcopts['lstm_decay_steps'] = 50000
tcopts['lstm_input'] = [0, 1, 2, 3, 6, 7]
tcopts['sampling_interval'] = 4
tcopts['lstm_num_input'] = len(tcopts['lstm_input'])
tcopts['lstm_num_classes'] = 2
tcopts['time_steps'] = 20
tcopts['start_frame'] = 20
tcopts['map_units'] = 4
tcopts['save_pos_thr'] = 0.83
tcopts['save_neg_thr'] = 0.83
tcopts['display_step'] = 50
