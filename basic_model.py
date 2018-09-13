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

    # 不支持传入最大长度
    # source_seq_length = 10  # source最大长度
    # target_seq_length = 8  # target最大长度


def get_multi_rnn_cell(rnn_size, num_layers):
    def lstm_cell():
        return tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    return tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])


class TextCNN(object):
    """文本分类，CNN模型"""
    def __init__(self, config):
        self.config = config

        # 输入
        self.source = tf.placeholder(tf.int32, [None, None], name='source')
        self.target = tf.placeholder(tf.int32, [None, None], name='target')
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        # self.is_train = tf.placeholder(tf.bool)

        # 1. encoder
        encoder_cell = get_multi_rnn_cell(self.config.rnn_size, self.config.num_layers)
        source_embedding = tf.get_variable('source_embedding', [self.config.source_vocab_size, self.config.encoding_embedding_size])
        source_embedding_inputs = tf.nn.embedding_lookup(source_embedding, self.source)
        encoder_output, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell,
            source_embedding_inputs,
            sequence_length=self.source_sequence_length,
            dtype=tf.float32)

        # 2. decoder
        # cut掉最后的字符
        ending = tf.strided_slice(self.target, [0, 0], [self.config.batch_size, -1], [1, 1])
        # 最前面加上<GO>字符
        decoder_input = tf.concat([tf.fill([tf.shape(self.target)[0], 1], self.config.target_letter_to_id['<GO>']), ending], 1)
        target_embedding = tf.get_variable('target_embedding', [self.config.target_vocab_size, self.config.decoding_embedding_size])
        target_embedding_inputs = tf.nn.embedding_lookup(target_embedding, decoder_input)

        # decoder_cell可以使用encoder_cell代替，即共享权重，但是不共享权重可以得到更佳的性能
        decoder_cell = get_multi_rnn_cell(self.config.rnn_size, self.config.num_layers)
        output_layer = Dense(self.config.target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 训练阶段
        # def train():
        #     with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_embedding_inputs, sequence_length=self.target_sequence_length)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            training_helper,
            encoder_state,  # # 使用encoder模块的输出状态来初始化states
            # 在输出添加full connected wrapper，映射得到每个target_vocab的score
            output_layer)
        training_decoder_output, _, __ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            impute_finished=True,  # 遇到EOS自动停止解码（EOS之后的所有time step的输出为0，输出状态为最后一个有效time step的输出状态）
            maximum_iterations=None)  # 设置最大decoding time steps数量，默认decode until the decoder is fully done，因为训练时会将target序列传入，所以可以为None
        self.logits = training_decoder_output.rnn_output

        # 测试阶段
        # def predict():
        #     with tf.variable_scope("decode", reuse=True):
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            target_embedding,
            tf.fill([tf.shape(self.source)[0]], self.config.target_letter_to_id['<GO>']),
            self.config.target_letter_to_id['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            predicting_helper,
            encoder_state,
            output_layer)
        predicting_decoder_output, _, __ = tf.contrib.seq2seq.dynamic_decode(
            predicting_decoder,
            impute_finished=True,  # 遇到EOS自动停止解码输出（停止输出，输出状态为最后一个有效time step的输出状态）
            maximum_iterations=tf.round(tf.reduce_max(self.source_sequence_length) * 2))  # 预测时不知道什么时候输出EOS，所以要设置最大time step数量
        self.result_ids = predicting_decoder_output.sample_id  # 输出target vocab id

        # 流程控制报错：ValueError: Outputs of true_fn and false_fn must have the same type: float32, int32
        # self.logits, self.result_ids = tf.cond(self.is_train, train, predict)

        # [<tf.Variable 'source_embedding:0' shape=(30, 15) dtype=float32_ref>, <tf.Variable 'rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:0' shape=(65, 200) dtype=float32_ref>, <tf.Variable 'rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0' shape=(200,) dtype=float32_ref>, <tf.Variable 'rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:0' shape=(100, 200) dtype=float32_ref>, <tf.Variable 'rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0' shape=(200,) dtype=float32_ref>, <tf.Variable 'target_embedding:0' shape=(30, 15) dtype=float32_ref>, <tf.Variable 'decoder/multi_rnn_cell/cell_0/lstm_cell/kernel:0' shape=(65, 200) dtype=float32_ref>, <tf.Variable 'decoder/multi_rnn_cell/cell_0/lstm_cell/bias:0' shape=(200,) dtype=float32_ref>, <tf.Variable 'decoder/multi_rnn_cell/cell_1/lstm_cell/kernel:0' shape=(100, 200) dtype=float32_ref>, <tf.Variable 'decoder/multi_rnn_cell/cell_1/lstm_cell/bias:0' shape=(200,) dtype=float32_ref>, <tf.Variable 'decoder/dense/kernel:0' shape=(50, 30) dtype=float32_ref>, <tf.Variable 'decoder/dense/bias:0' shape=(30,) dtype=float32_ref>]
        print(tf.trainable_variables())

        # 3. optimize
        masks = tf.sequence_mask(self.target_sequence_length, tf.reduce_max(self.target_sequence_length), dtype=tf.float32)
        # logits:[batch_size, 10, 30], targets:[batch_size, 10], masks:[batch_size, 10]
        # 让mask的内容不计算损失，如果不做mask，即使impute_finished=True使得EOS之后的输出为0，但是targets的PAD会被误认为是一个正常的翻译结果而计算了损失
        self.loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.target, masks)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        # 梯度裁剪
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)
