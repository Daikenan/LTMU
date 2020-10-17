import tensorflow as tf
from tensorflow.contrib import rnn
from tcopt import tcopts
class tcnet:
    def __init__(self):
        self.learningRates = {}

    def net(self, inputs, reuse=False):
        with tf.variable_scope('tcnet') as scope:
            if reuse:
                scope.reuse_variables()
            with tf.variable_scope('conv1'):
                weights = tf.get_variable("weights", shape=(7, 7, 3, 96), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01), trainable=False)
                biases = tf.get_variable("biases", shape=(96,), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0), trainable=False)
                outputs = tf.nn.conv2d(inputs, weights, strides=[1, 2, 2, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)
                outputs = tf.nn.lrn(outputs, depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
                outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            with tf.variable_scope('conv2'):
                weights = tf.get_variable("weights", shape=(5, 5, 96, 256), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01), trainable=False)
                biases = tf.get_variable("biases", shape=(256,), dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                                         trainable=False)
                outputs = tf.nn.conv2d(outputs, weights, strides=[1, 2, 2, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)
                outputs = tf.nn.lrn(outputs, depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
                outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

            with tf.variable_scope('conv3'):
                weights = tf.get_variable("weights", shape=(3, 3, 256, 512), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01), trainable=False)
                biases = tf.get_variable("biases", shape=(512,), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0), trainable=False)
                outputs3 = tf.nn.conv2d(outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs3)

            with tf.variable_scope('conv4'):
                weights = tf.get_variable("weights", shape=(3, 3, 512, 512), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01),
                                          regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                biases = tf.get_variable("biases", shape=(512,), dtype=tf.float32, initializer=tf.constant_initializer(0.0),
                                         regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                outputs = tf.nn.dropout(outputs,keep_prob=0.5, seed=1)
                outputs = tf.nn.conv2d(outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)

            with tf.variable_scope('fc5'):
                outputs = tf.contrib.layers.flatten(outputs)
                weights = tf.get_variable("weights", shape=(512, 512), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                biases = tf.get_variable("biases", shape=(512,), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0),
                                         regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                outputs = tf.nn.dropout(outputs,keep_prob=0.5, seed=1)
                outputs = tf.matmul(outputs, weights) + biases
                outputs = tf.nn.relu(outputs)

            with tf.variable_scope('fc6'):
                weights = tf.get_variable("weights", shape=(512, 2), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                biases = tf.get_variable("biases", shape=(2,), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0),
                                         regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                outputs = tf.nn.dropout(outputs,keep_prob=0.5, seed=1)
                outputs = tf.matmul(outputs, weights) + biases

        return outputs

    def extractFeature(self, inputs):
        with tf.variable_scope('tcnet') as scope:
            scope.reuse_variables()
            with tf.variable_scope('conv1'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.conv2d(inputs, weights, strides=[1, 2, 2, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)
                outputs = tf.nn.lrn(outputs, depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
                outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            with tf.variable_scope('conv2'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.conv2d(outputs, weights, strides=[1, 2, 2, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs)
                outputs = tf.nn.lrn(outputs, depth_radius=2, bias=1.0, alpha=0.0001, beta=0.75)
                outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            with tf.variable_scope('conv3'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs3 = tf.nn.conv2d(outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs3)

        return outputs

    def classification(self, inputs):
        with tf.variable_scope('tcnet') as scope:
            scope.reuse_variables()
            with tf.variable_scope('conv4'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.dropout(inputs, keep_prob=0.5, seed=1)
                outputs4 = tf.nn.conv2d(outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                outputs = tf.nn.relu(outputs4)
            with tf.variable_scope('fc5'):
                outputs = tf.contrib.layers.flatten(outputs)
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.dropout(outputs, keep_prob=0.5, seed=1)
                outputs = tf.matmul(outputs, weights) + biases
                outputs = tf.nn.relu(outputs)
            with tf.variable_scope('fc6'):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")
                outputs = tf.nn.dropout(outputs, keep_prob=0.5, seed=1)
                outputs = tf.matmul(outputs, weights) + biases
        score = tf.nn.softmax(outputs, dim=1)
        return outputs, score, outputs4

    def loss(self, inputs, label):

        with tf.variable_scope('tcnet') as scope:
            scope.reuse_variables()
            loss1 = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=inputs)
            loss = tf.reduce_sum(loss1)
            with tf.variable_scope('conv4'):
                weights1 = tf.get_variable('weights')
            with tf.variable_scope('fc5'):
                weights2 = tf.get_variable('weights')
            with tf.variable_scope('fc6'):
                weights3 = tf.get_variable('weights')
            loss += (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3)) * 5e-4
        return loss, loss1

class tclstm:
    def net(self, inputs, reuse=False):
        with tf.variable_scope('tclstm') as scope:
            if reuse:
                scope.reuse_variables()
            with tf.variable_scope('lstm_layer1'):

                # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
                x = tf.unstack(inputs, tcopts['time_steps'], 1)

                # Define a lstm cell with tensorflow
                lstm_cell = rnn.BasicLSTMCell(64, forget_bias=1.0)

                # Get lstm cell output
                outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
            with tf.variable_scope('lstm_layer2'):
                lstm_cell = rnn.BasicLSTMCell(64, forget_bias=1.0)
                outputs, states = rnn.static_rnn(lstm_cell, outputs[-8:], dtype=tf.float32, initial_state=states)
            with tf.variable_scope('lstm_layer3'):
                lstm_cell = rnn.BasicLSTMCell(64, forget_bias=1.0)
                outputs, states = rnn.static_rnn(lstm_cell, outputs[-3:], dtype=tf.float32, initial_state=states)

            with tf.variable_scope('fc1'):
                weights = tf.get_variable("weights", shape=(64, 64), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                biases = tf.get_variable("biases", shape=(64,), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0),
                                         regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                outputs = tf.matmul(outputs[-1], weights) + biases
            with tf.variable_scope('fc2'):
                weights = tf.get_variable("weights", shape=(64, 2), dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                biases = tf.get_variable("biases", shape=(2,), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0),
                                         regularizer=tf.contrib.layers.l2_regularizer(5e-4), trainable=True)
                outputs = tf.matmul(outputs, weights) + biases
        return outputs

#
# W = 107
# H = 107
# C = 3
# tcinput = tf.placeholder(tf.float32, [1, H, W, 3])
# a = tcnet()
# tcoutput = a.net(tcinput)
# with tf.Session() as sess:
#     summary_writer = tf.summary.FileWriter('./log/', tf.get_default_graph())
#
# print('ok')

