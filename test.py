# -*- coding: utf-8 -*-

"""This file is just for testing"""

import tensorflow as tf
import numpy as np

# target = tf.placeholder(tf.int32, [None, None], name='target')
# ending = tf.strided_slice(target, [0, 0], [2, -1], [1, 1])  # cut掉最后的字符
# decoder_input = tf.concat([tf.fill([2, 1], 88), ending], 1)  # 最前面加上<GO>字符
#
# with tf.Session() as sess:
#     X = [[1, 2], [3, 4]]
#     res = sess.run(decoder_input, feed_dict={target: X})
#     print(res)  # [[88  1], [88  3]]


# res = tf.sequence_mask([1, 3, 2], 5, dtype=tf.float32)
# with tf.Session() as sess:
#     print(sess.run(res))  # [[1. 0. 0. 0. 0.],[1. 1. 1. 0. 0.],[1. 1. 0. 0. 0.]]


# W = tf.get_variable(shape=(5,), name="W")
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())  # 只全局初始化一次
#     print(sess.run(W))  # [ 0.65224755 -0.75201756 -0.2431001   0.24485707 -0.4320725 ]
#     print(sess.run(W))  # [ 0.65224755 -0.75201756 -0.2431001   0.24485707 -0.4320725 ]
#
#     sess.run(tf.global_variables_initializer())  # 重新初始化
#     print(sess.run(W))  # [-0.18968838  0.6756437  -0.5558038  -0.3496107  -0.04675609]


# x = tf.placeholder(tf.float32, [None, 28, 28, 3])
# conv1 = tf.contrib.layers.conv2d(inputs=x, num_outputs=32, kernel_size=(2, 2))
# conv2 = tf.contrib.layers.conv2d(inputs=x, num_outputs=32, kernel_size=(2, 2))  # 不报错，系统自行加上并检测variable scope
# print(tf.trainable_variables())  # [<tf.Variable 'Conv/weights:0' shape=(2, 2, 3, 32) dtype=float32_ref>,<tf.Variable 'Conv_1/weights:0' shape=(2, 2, 3, 32) dtype=float32_ref>]


# n_inputs = 64
# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
# basic_cell1 = tf.contrib.rnn.BasicRNNCell(num_units=8)
# basic_cell2 = tf.contrib.rnn.BasicRNNCell(num_units=8)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell1, [X0, X1], dtype=tf.float32)  # 系统自行加上variable scope
# output_seqs1, states1 = tf.contrib.rnn.static_rnn(basic_cell1, [X0, X1], dtype=tf.float32)  # 同一个cell给两个static_rnn使用，与reuse无关
# # output_seqs2, states2 = tf.contrib.rnn.static_rnn(basic_cell2, [X0, X1], dtype=tf.float32)  # 报错，ValueError: Variable rnn/basic_rnn_cell/kernel already exists, disallowed.
# print(tf.trainable_variables())  # [<tf.Variable 'rnn/basic_rnn_cell/kernel:0' shape=(72, 8) dtype=float32_ref>]

# 同上
# def get_multi_rnn_cell(rnn_size, num_layers):
#     def lstm_cell():
#         return tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
#
#     return tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
#
#
# encoder_cell = get_multi_rnn_cell(50, 2)
# encoder_cell1 = get_multi_rnn_cell(50, 2)
# encoder_output, encoder_state = tf.nn.dynamic_rnn(
#     encoder_cell,
#     tf.placeholder(tf.float32, [1, 5, 64]),
#     sequence_length=[5],
#     dtype=tf.float32)
# encoder_output1, encoder_state1 = tf.nn.dynamic_rnn(
#     encoder_cell1,
#     tf.placeholder(tf.float32, [1, 5, 64]),
#     sequence_length=[5],
#     dtype=tf.float32)
# print(tf.trainable_variables())

# [
# <tf.Variable 'source_embedding:0' shape=(30, 15) dtype=float32_ref>,
# <tf.Variable 'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0' shape=(65, 200) dtype=float32_ref>,
# <tf.Variable 'rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0' shape=(200,) dtype=float32_ref>,
# <tf.Variable 'rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0' shape=(100, 200) dtype=float32_ref>,
# <tf.Variable 'rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0' shape=(200,) dtype=float32_ref>
# <tf.Variable 'target_embedding:0' shape=(30, 15) dtype=float32_ref>
# <tf.Variable 'decoder/multi_rnn_cell/cell_0/lstm_cell/kernel:0' shape=(65, 200) dtype=float32_ref>
# <tf.Variable 'decoder/multi_rnn_cell/cell_0/lstm_cell/bias:0' shape=(200,) dtype=float32_ref>
# <tf.Variable 'decoder/multi_rnn_cell/cell_1/lstm_cell/kernel:0' shape=(100, 200) dtype=float32_ref>
# <tf.Variable 'decoder/multi_rnn_cell/cell_1/lstm_cell/bias:0' shape=(200,) dtype=float32_ref>
# <tf.Variable 'decoder/dense/kernel:0' shape=(50, 30) dtype=float32_ref>
# <tf.Variable 'decoder/dense/bias:0' shape=(30,) dtype=float32_ref>
# ]


# go_ = np.array([[0] * 2]).transpose()
# print(go_)
# padded = [[1, 2, 3], [4, 5, 6]]
# cuted_ = np.array(padded)[:, 0:len(padded[0]) - 1]
# print(np.hstack((go_, cuted_)))
