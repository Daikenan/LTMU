from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from tcNet import tclstm
import os
from tcopt import tcopts
import tensorflow.contrib.slim as slim
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def prepare_test_data(Dataset, seq_len, mode=None):
    base_dir = tcopts['train_data_dir']
    data_dir = os.path.join(base_dir, Dataset)
    train_list = os.listdir(data_dir)
    train_list.sort()
    np.random.shuffle(train_list)

    testing_set_dir = '../../utils/testing_set.txt'
    testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test':
        print('test data')
        train_list = [vid for vid in train_list if vid[:-4] in testing_set]
    elif mode == 'train':
        print('train data')
        train_list = [vid for vid in train_list if vid[:-4] not in testing_set]
    else:
        print("all data")
    pos_data, neg_data = prepare_data(data_dir, seq_len, train_list)
    np.save('test_neg_data.npy', np.array(neg_data))
    np.save('test_pos_data.npy', np.array(pos_data))
    return pos_data, neg_data

def prepare_train_data(Dataset, seq_len, mode=None):
    base_dir = tcopts['train_data_dir']
    data_dir = os.path.join(base_dir, Dataset)
    train_list = os.listdir(data_dir)
    train_list.sort()
    np.random.shuffle(train_list)

    testing_set_dir = '../../utils/testing_set.txt'
    testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test':
        print('test data')
        train_list = [vid for vid in train_list if vid[:-4] in testing_set and vid.endswith('.txt')]
    elif mode == 'train':
        print('train data')
        train_list = [vid for vid in train_list if vid[:-4] not in testing_set and vid.endswith('.txt')]
    else:
        print("all data")

    pos_data, neg_data = prepare_data(data_dir, seq_len,train_list)
    np.save('neg_data.npy', np.array(neg_data))
    np.save('pos_data.npy', np.array(pos_data))
    return pos_data, neg_data

def prepare_data(data_dir, seq_len, train_list):


    pos_data = []
    neg_data = []
    sampling_interval = tcopts['sampling_interval']
    # video
    for id, video in enumerate(train_list):
        print(str(id) + ':' + video)
        txt_tmp = np.loadtxt(os.path.join(data_dir, train_list[id]), delimiter=',')
        map_tmp = np.load(os.path.join(data_dir, train_list[id][:-4]+'_map.npy'))
        loss_list = np.where(txt_tmp[:, 5] == 0)[0]
        for i in range((len(txt_tmp) - seq_len)//sampling_interval):
            if sampling_interval * i + seq_len + 1 >= len(txt_tmp):
                continue
            data_tmp = txt_tmp[sampling_interval*i+1:sampling_interval*i + seq_len+1]
            loss_index = np.concatenate([np.where(data_tmp[:, 5] == -1)[0], np.where(data_tmp[:, 5] == 0)[0]])
            if data_tmp[-1, 5] > tcopts['pos_thr']:
                # pos data
                pos_data.append([data_tmp, train_list[id][:-4]+'_map.npy'])
            elif data_tmp[-1, 5] == 0:
                neg_data.append([data_tmp, train_list[id][:-4]+'_map.npy'])
    return pos_data, neg_data

def get_batch_input(pos_data, neg_data, batch_size):

    pos_id = np.random.randint(0, len(pos_data), batch_size)
    neg_id = np.random.randint(0, len(neg_data), batch_size)
    pos_map = []
    neg_map = []

    for i in range(len(pos_data[pos_id])):
        map_tmp = np.load(os.path.join(tcopts['train_data_dir'], pos_data[pos_id][i][1]), allow_pickle=True)
        frame_index = pos_data[pos_id][i][0][:, 4]
        map_tmp = np.reshape(map_tmp[np.array(frame_index, dtype='int16') - 1], [tcopts['time_steps'], 1, 19, 19])
        map_tmp = map_tmp.transpose((0, 2, 3, 1))
        pos_map.append(map_tmp)
    for i in range(len(neg_data[neg_id])):
        map_tmp = np.load(os.path.join(tcopts['train_data_dir'], neg_data[neg_id][i][1]), allow_pickle=True)
        frame_index = neg_data[neg_id][i][0][:, 4]
        map_tmp = np.reshape(map_tmp[np.array(frame_index, dtype='int16') - 1], [tcopts['time_steps'], 1, 19, 19])
        map_tmp = map_tmp.transpose((0, 2, 3, 1))
        neg_map.append(map_tmp)
    batch_X_input = np.concatenate((pos_data[pos_id][:, 0], neg_data[neg_id][:, 0]), axis=0)
    batch_X_input = np.array(list(batch_X_input))
    batch_map_input = np.concatenate((pos_map, neg_map), axis=0)
    labels1 = np.array([0, 1]).reshape((1, 2))
    labels1 = np.tile(labels1, (batch_size, 1))
    labels2 = np.array([1, 0]).reshape((1, 2))
    labels2 = np.tile(labels2, (batch_size, 1))
    labels = np.concatenate((labels1, labels2), axis=0)
    batch_map_input = np.reshape(batch_map_input,
                                 [batch_map_input.shape[0] * batch_map_input.shape[1], batch_map_input.shape[2],
                                  batch_map_input.shape[3],
                                  batch_map_input.shape[4]])
    return batch_X_input, batch_map_input, labels
def get_test_batch_input(pos_data, neg_data):
    batch_size = 1000
    pos_id = np.random.randint(0, len(pos_data), batch_size)
    neg_id = np.random.randint(0, len(neg_data), batch_size)
    pos_map = []
    neg_map = []

    for i in range(len(pos_data[pos_id])):
        map_tmp = np.load(os.path.join(tcopts['train_data_dir'], pos_data[pos_id][i][1]), allow_pickle=True)
        frame_index = pos_data[pos_id][i][0][:, 4]
        map_tmp = np.reshape(map_tmp[np.array(frame_index, dtype='int16') - 1], [tcopts['time_steps'], 1, 19, 19])
        map_tmp = map_tmp.transpose((0, 2, 3, 1))
        pos_map.append(map_tmp)
    for i in range(len(neg_data[neg_id])):
        map_tmp = np.load(os.path.join(tcopts['train_data_dir'], neg_data[neg_id][i][1]), allow_pickle=True)
        frame_index = neg_data[neg_id][i][0][:, 4]
        map_tmp = np.reshape(map_tmp[np.array(frame_index, dtype='int16') - 1], [tcopts['time_steps'], 1, 19, 19])
        map_tmp = map_tmp.transpose((0, 2, 3, 1))
        neg_map.append(map_tmp)
    batch_X_input = np.concatenate((pos_data[pos_id][:, 0], neg_data[neg_id][:, 0]), axis=0)
    batch_X_input = np.array(list(batch_X_input))
    batch_map_input = np.concatenate((pos_map, neg_map), axis=0)
    labels1 = np.array([0, 1]).reshape((1, 2))
    labels1 = np.tile(labels1, (batch_size, 1))
    labels2 = np.array([1, 0]).reshape((1, 2))
    labels2 = np.tile(labels2, (batch_size, 1))
    labels = np.concatenate((labels1, labels2), axis=0)
    batch_map_input = np.reshape(batch_map_input,
                                 [batch_map_input.shape[0] * batch_map_input.shape[1], batch_map_input.shape[2],
                                  batch_map_input.shape[3],
                                  batch_map_input.shape[4]])

    return len(pos_data[pos_id]), len(neg_data[neg_id]), batch_X_input, batch_map_input, labels

def load_training_data(pos_name, neg_name):
    pos_data = np.load(pos_name, allow_pickle=True)
    neg_data = np.load(neg_name, allow_pickle=True)
    return pos_data, neg_data

def train():
    model = tclstm()
    training_steps = 50000
    display_step = tcopts['display_step']

    with tf.device('cpu:0'):
        global_step = slim.create_global_step()
        # tf Graph input
        maps = tf.placeholder("float", [None, 19, 19, 1])
        map_logits = model.map_net(maps)
        X = tf.placeholder("float", [None, tcopts['time_steps'], tcopts['lstm_num_input']])
        Inputs = tf.concat((X, map_logits), axis=2)
        Y = tf.placeholder("float", [None, tcopts['lstm_num_classes']])
        lrOp = tf.train.exponential_decay(tcopts['lstm_initial_lr'],
                                          global_step,
                                          tcopts['lstm_decay_steps'],
                                          tcopts['lr_decay_factor'],
                                          staircase=True)
        logits, _ = model.net(Inputs)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrOp)
        optimizer = tf.train.MomentumOptimizer(learning_rate=lrOp, momentum=0.9)
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        prediction = tf.nn.softmax(logits)
        grads = optimizer.compute_gradients(loss_op)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        test_accuracy = accuracy

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        pos_data, neg_data = load_training_data('pos_data.npy', 'neg_data.npy')
        test_pos_data, test_neg_data = load_training_data('test_pos_data.npy', 'test_neg_data.npy')
        test_pos_num, test_neg_num, test_batch_X_input, test_batch_map_input, test_labels = get_test_batch_input(
            test_pos_data, test_neg_data)

        test_accuracy_pos = tf.reduce_mean(tf.cast(correct_pred[:test_pos_num], tf.float32))
        test_accuracy_neg = tf.reduce_mean(tf.cast(correct_pred[test_pos_num:], tf.float32))
        accuracy_pos = tf.reduce_mean(tf.cast(correct_pred[:tcopts['batch_size']], tf.float32))
        accuracy_neg = tf.reduce_mean(tf.cast(correct_pred[tcopts['batch_size']:], tf.float32))

        saver = tf.train.Saver(keep_checkpoint_every_n_hours=tcopts['keep_checkpoint_every_n_hours'])
        # add summary
        tf.summary.scalar('learning_rate', lrOp)
        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('training_accuracy', accuracy)
        tf.summary.scalar('training_accuracy_pos', accuracy_pos)
        tf.summary.scalar('training_accuracy_neg', accuracy_neg)
        # grads
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        # # trainable var
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        summary_op = tf.summary.merge_all()
        tf.summary.scalar('testing_accuracy_pos', test_accuracy_pos)
        tf.summary.scalar('testing_accuracy_neg', test_accuracy_neg)
        tf.summary.scalar('testing_accuracy', test_accuracy)
        test_merge_summary = tf.summary.merge([tf.get_collection(tf.GraphKeys.SUMMARIES, 'testing_accuracy')])

        summary_writer = tf.summary.FileWriter(tcopts['lstm_train_dir'], graph=tf.get_default_graph())
        tfconfig = tf.ConfigProto()
        # tfconfig.gpu_options.per_process_gpu_memory_fraction = 1
        # Start training
        with tf.Session(config=tfconfig) as sess:

            # Run the initializer
            sess.run(init)
            checkpoint = tf.train.latest_checkpoint(tcopts['lstm_train_dir'])
            if checkpoint is not None:
                saver.restore(sess, checkpoint)
            best_model = (tcopts['save_neg_thr']+tcopts['save_pos_thr'])/2
            while True:
                batch_X_input, batch_map_input, labels = get_batch_input(pos_data, neg_data, tcopts['batch_size'])
                # Reshape data to get 28 seq of 28 elements
                # Run optimization op (backprop)
                _, g_step = sess.run([train_op, global_step], feed_dict={X: batch_X_input[:, :, tcopts['lstm_input']], Y: labels, maps: batch_map_input})
                if g_step % display_step == 0:
                    # Calculate batch loss and accuracy
                    loss, acc, acc_pos, acc_neg, summary_str = sess.run(
                        [loss_op, accuracy, accuracy_pos, accuracy_neg, summary_op],
                        feed_dict={X: batch_X_input[:, :, tcopts['lstm_input']],
                                   Y: labels,
                                   maps: batch_map_input})
                    summary_writer.add_summary(summary_str, g_step)
                    print("Step " + str(g_step) + ", Minibatch Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc) + ", Accuracy pos= " + "{:.3f}".format(
                        acc_pos) + ", Accuracy neg= " + "{:.3f}".format(acc_neg))

                if g_step % tcopts['model_save_interval'] == 0:
                    checkpoint_path = os.path.join(tcopts['lstm_train_dir'], 'lstm_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=g_step)
                    print('Save model, global step: %d' % g_step)
                if g_step % tcopts['eval_interval'] == 0:
                    test_pos_num, test_neg_num, test_batch_X_input, test_batch_map_input, test_labels = get_test_batch_input(
                        test_pos_data, test_neg_data)
                    test_acc, test_acc_pos, test_acc_neg, test_summary_str = sess.run(
                        [test_accuracy, test_accuracy_pos, test_accuracy_neg, test_merge_summary],
                        feed_dict={X: test_batch_X_input[:, :, tcopts['lstm_input']], Y: test_labels, maps: test_batch_map_input})
                    summary_writer.add_summary(test_summary_str, g_step)
                    print("test accuracy:" + "{:.4f}".format(test_acc) + "  test accuracy pos:" + "{:.4f}".format(
                        test_acc_pos) + "  test accuracy neg:" + "{:.4f}".format(test_acc_neg))
                    if test_acc_pos > tcopts['save_pos_thr'] and test_acc_neg > tcopts['save_neg_thr'] and (test_acc_pos+test_acc_neg)/2>best_model:
                        best_model = (test_acc_pos+test_acc_neg)/2
                        checkpoint_path = os.path.join(tcopts['lstm_train_dir'], 'lstm_model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=g_step)
                        print('Save model, global step: %d' % g_step)














if __name__ == '__main__':
    prepare_train_data('', tcopts['time_steps'], mode='train')
    prepare_test_data('', tcopts['time_steps'], mode='test')
    train()

