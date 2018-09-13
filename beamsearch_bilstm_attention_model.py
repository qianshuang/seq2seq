# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense


class TCNNConfig(object):
    """CNN配置参数"""
    print_per_batch = 10    # 每多少轮输出一次结果
    num_epochs = 1000
    batch_size = 128
    rnn_size = 50
    num_layers = 2
    encoding_embedding_size = 15
    decoding_embedding_size = 15
    learning_rate = 0.001
    beam_width = 5


def get_multi_rnn_cell(rnn_size, num_layers):
    return tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 输入
        self.source = tf.placeholder(tf.int32, [None, None], name='source')
        self.target = tf.placeholder(tf.int32, [None, None], name='target')
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        # 1. encoder
        source_embedding = tf.get_variable('source_embedding', [self.config.source_vocab_size, self.config.encoding_embedding_size])
        source_embedding_inputs = tf.nn.embedding_lookup(source_embedding, self.source)

        # bi-LSTM
        fw_lstm_cell = tf.contrib.rnn.LSTMCell(int(self.config.rnn_size / 2), initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        bw_lstm_cell = tf.contrib.rnn.LSTMCell(int(self.config.rnn_size / 2), initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        (outputs, (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm_cell,
                                                                          cell_bw=bw_lstm_cell,
                                                                          inputs=source_embedding_inputs,
                                                                          sequence_length=self.source_sequence_length,
                                                                          dtype=tf.float32)
        encoder_output = tf.concat(outputs, -1)
        encoder_final_state_c = tf.concat([fw_state.c, bw_state.c], -1)
        encoder_final_state_h = tf.concat([fw_state.h, bw_state.h], -1)
        encoder_state = tf.contrib.rnn.LSTMStateTuple(
            c=encoder_final_state_c,
            h=encoder_final_state_h
        )

        # 2. decoder
        ending = tf.strided_slice(self.target, [0, 0], [self.config.batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([tf.shape(self.target)[0], 1], self.config.target_letter_to_id['<GO>']), ending], 1)
        target_embedding = tf.get_variable('target_embedding', [self.config.target_vocab_size, self.config.decoding_embedding_size])
        target_embedding_inputs = tf.nn.embedding_lookup(target_embedding, decoder_input)

        decoder_cell = get_multi_rnn_cell(self.config.rnn_size, self.config.num_layers)
        output_layer = Dense(self.config.target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 训练阶段
        with tf.variable_scope("decode"):
            # attention
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.rnn_size, encoder_output, memory_sequence_length=self.source_sequence_length)
            attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=self.config.rnn_size)

            # initial_state
            initial_state = attention_decoder_cell.zero_state(tf.shape(self.source)[0], tf.float32).clone(cell_state=encoder_state)

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_embedding_inputs, sequence_length=self.target_sequence_length)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                attention_decoder_cell,
                training_helper,
                initial_state,  # 使用encoder模块的输出状态来初始化attention_decoder的初始states，若直接使用encoder_state会报错
                output_layer)
            training_decoder_output, _, __ = tf.contrib.seq2seq.dynamic_decode(
                training_decoder,
                impute_finished=True,  # 遇到EOS自动停止解码（EOS之后的所有time step的输出为0，输出状态为最后一个有效time step的输出状态）
                maximum_iterations=None)  # 设置最大decoding time steps数量，默认decode until the decoder is fully done，因为训练时会将target序列传入，所以可以为None
            self.logits = training_decoder_output.rnn_output

        # 测试阶段
        with tf.variable_scope("decode", reuse=True):
            # a. decoder_attention
            bs_encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=self.config.beam_width)  # tile_batch等价于复制10份，然后concat(..., 0)
            bs_sequence_length = tf.contrib.seq2seq.tile_batch(self.source_sequence_length, multiplier=self.config.beam_width)

            bs_attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.rnn_size, bs_encoder_output, memory_sequence_length=bs_sequence_length)
            bs_attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, bs_attention_mechanism, attention_layer_size=self.config.rnn_size)

            # b. decoder_initial_state
            bs_cell_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.config.beam_width)
            bs_initial_state = bs_attention_decoder_cell.zero_state(tf.shape(self.source)[0] * self.config.beam_width, tf.float32).clone(cell_state=bs_cell_state)

            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=bs_attention_decoder_cell,
                embedding=target_embedding,
                start_tokens=tf.fill([tf.shape(self.source)[0]], self.config.target_letter_to_id['<GO>']),
                end_token=self.config.target_letter_to_id['<EOS>'],
                initial_state=bs_initial_state,
                beam_width=self.config.beam_width,
                output_layer=output_layer,
                length_penalty_weight=0.0)  # 对长度较短的生成结果施加惩罚，0.0表示不惩罚

            predicting_decoder_output, _, __ = tf.contrib.seq2seq.dynamic_decode(
                predicting_decoder,
                impute_finished=False,  # 遇到EOS自动停止解码输出（停止输出，输出状态为最后一个有效time step的输出状态）
                maximum_iterations=tf.round(tf.reduce_max(self.source_sequence_length) * 2))  # 预测时不知道什么时候输出EOS，所以要设置最大time step数量
            self.result_ids = tf.transpose(predicting_decoder_output.predicted_ids, perm=[0, 2, 1])  # 输出target vocab id：[batch_size, beam_width, max_time_step]

        print(tf.trainable_variables())

        # 3. optimize
        masks = tf.sequence_mask(self.target_sequence_length, tf.reduce_max(self.target_sequence_length), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.target, masks)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        # 梯度裁剪
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)
