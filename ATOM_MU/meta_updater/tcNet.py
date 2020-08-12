import tensorflow as tf
from tensorflow.contrib import rnn
from tcopt import tcopts


class tclstm:
    def map_net(self, maps, reuse=False):
        with tf.variable_scope('mapnet') as scope:
            if reuse:
                scope.reuse_variables()
            with tf.variable_scope('conv1'):
                weights = tf.get_variable("weights", shape=(3, 3, 1, 32), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01), trainable=True)
                biases = tf.get_variable("biases", shape=(32,), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0), trainable=True)
                map_outputs = tf.nn.conv2d(maps, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                map_outputs = tf.nn.relu(map_outputs)
                map_outputs = tf.nn.max_pool(map_outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            with tf.variable_scope('conv2'):
                weights = tf.get_variable("weights", shape=(3, 3, 32, 64), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01), trainable=True)
                biases = tf.get_variable("biases", shape=(64,), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0), trainable=True)
                map_outputs = tf.nn.conv2d(map_outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                map_outputs = tf.nn.relu(map_outputs)
            with tf.variable_scope('conv3'):
                weights = tf.get_variable("weights", shape=(1, 1, 64, tcopts['map_units']), dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(0.01), trainable=True)
                biases = tf.get_variable("biases", shape=(tcopts['map_units'],), dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.0), trainable=True)
                map_outputs = tf.nn.conv2d(map_outputs, weights, strides=[1, 1, 1, 1], padding='VALID') + biases
                map_outputs = tf.nn.relu(map_outputs)
                # GAP
                map_outputs = tf.reduce_mean(map_outputs, axis=[1, 2])
                map_outputs = tf.reshape(map_outputs, [-1, tcopts['time_steps'], tcopts['map_units']])
        return map_outputs

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
                DQN_Inputs = outputs
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
        return outputs, DQN_Inputs



