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


# 如果是用bi-LSTM做encoder，就不能再用multi_rnn_cell做decoder了，也不能用bi-LSTM做decoder，只能使用单层的RNN单元
def get_multi_rnn_cell(rnn_size, num_layers):
    # def lstm_cell():
    #     return tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    # return tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
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

        # attention
        decoder_cell = get_multi_rnn_cell(self.config.rnn_size, self.config.num_layers)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.config.rnn_size, encoder_output, memory_sequence_length=self.source_sequence_length)
        attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=self.config.rnn_size)
        # initial_state
        initial_state = attention_decoder_cell.zero_state(tf.shape(self.source)[0], tf.float32).clone(cell_state=encoder_state)

        output_layer = Dense(self.config.target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 训练阶段
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
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            target_embedding,
            tf.fill([tf.shape(self.source)[0]], self.config.target_letter_to_id['<GO>']),
            self.config.target_letter_to_id['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            attention_decoder_cell,
            predicting_helper,
            initial_state,
            output_layer)
        predicting_decoder_output, _, __ = tf.contrib.seq2seq.dynamic_decode(
            predicting_decoder,
            impute_finished=True,  # 遇到EOS自动停止解码输出（停止输出，输出状态为最后一个有效time step的输出状态）
            maximum_iterations=tf.round(tf.reduce_max(self.source_sequence_length) * 2))  # 预测时不知道什么时候输出EOS，所以要设置最大time step数量
        self.result_ids = predicting_decoder_output.sample_id  # 输出target vocab id

        # [<tf.Variable 'memory_layer/kernel:0' shape=(50, 50) dtype=float32_ref>,<tf.Variable 'decoder/attention_wrapper/attention_layer/kernel:0' shape=(100, 50) dtype=float32_ref>]
        print(tf.trainable_variables())  # 上面分别是计算Attention score的参数W和计算Attention vector的参数Wc，其他和basic模型一样

        # 3. optimize
        masks = tf.sequence_mask(self.target_sequence_length, tf.reduce_max(self.target_sequence_length), dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.target, masks)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        # 梯度裁剪
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)
