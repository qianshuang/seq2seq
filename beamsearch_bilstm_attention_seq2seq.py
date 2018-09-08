import numpy as np
# import time
import tensorflow as tf
from tensorflow.python.layers.core import Dense

with open('data/letters_source.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()

with open('data/letters_target.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()


def extract_character_vocab(data):
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set([character for line in data.split('\n') for character in line]))
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int


source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data.split('\n')]
target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]


def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


def get_encoder_layer(input_data, rnn_size, num_layers, source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    # bi-LSTM cell 注：bi-LSTM只需要用在encoder阶段
    fw_lstm_cell = tf.contrib.rnn.LSTMCell(int(rnn_size / 2),
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    bw_lstm_cell = tf.contrib.rnn.LSTMCell(int(rnn_size / 2),
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    (outputs, (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm_cell,
                                                                      cell_bw=bw_lstm_cell,
                                                                      inputs=encoder_embed_input,
                                                                      sequence_length=source_sequence_length,
                                                                      dtype=tf.float32)
    encoder_output = tf.concat(outputs, -1)
    encoder_final_state_c = tf.concat([fw_state.c, bw_state.c], -1)
    encoder_final_state_h = tf.concat([fw_state.h, bw_state.h], -1)
    encoder_state = tf.contrib.rnn.LSTMStateTuple(
        c=encoder_final_state_c,
        h=encoder_final_state_h
    )

    return encoder_output, encoder_state


def process_decoder_input(data, vocab_to_int, batch_size):
    # cut掉最后的<EOS>字符
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    # 最前面加上<GO>字符
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return decoder_input


def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size, target_sequence_length,
                   max_target_sequence_length, encoder_state, decoder_input, attention_decoder_cell, encoder_outputs):
    # 1. Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 2. initial_state
    initial_state = attention_decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

    # 3. Output全连接层
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # 4. Training decoder
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length, time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(attention_decoder_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer)
        training_decoder_output, _, __ = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True,
                                                                           maximum_iterations=max_target_sequence_length)

    # 5. Predicting decoder
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')

        # tilt_encoder_state = tf.contrib.rnn.LSTMStateTuple(tf.contrib.seq2seq.tile_batch(encoder_state[0], multiplier=10),
        #                                           tf.contrib.seq2seq.tile_batch(encoder_state[1], multiplier=10)),
        bs_cell_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=1)  # beam_width = 10
        decoder_initial_state = attention_decoder_cell.zero_state(batch_size * 1, tf.float32).clone(cell_state=bs_cell_state)

        encoder_outputs_beam = tf.contrib.seq2seq.tile_batch(encoder_outputs, 1)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(rnn_size, encoder_outputs_beam, memory_sequence_length=source_sequence_length)
        # LSTM cell
        cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=rnn_size)

        predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=attention_decoder_cell,
            embedding=decoder_embeddings,
            start_tokens=start_tokens,
            end_token=target_letter_to_int['<EOS>'],
            initial_state=decoder_initial_state,
            beam_width=1,
            output_layer=output_layer,
            length_penalty_weight=0.0)

        predicting_decoder_output, _, __ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                             impute_finished=True,  # 遇到EOS自动停止解码
                                                                             maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output


def seq2seq_model(input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers):
    encoder_outputs, encoder_state = get_encoder_layer(input_data,
                                                       rnn_size,
                                                       num_layers,
                                                       source_sequence_length,
                                                       source_vocab_size,
                                                       encoding_embedding_size)
    # Attention
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(rnn_size, encoder_outputs,
                                                            memory_sequence_length=source_sequence_length)
    # LSTM cell
    cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism,
                                                                 attention_layer_size=rnn_size)
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input,
                                                                        attention_decoder_cell,
                                                                        encoder_outputs)
    return training_decoder_output, predicting_decoder_output


epochs = 200
batch_size = 128
rnn_size = 50
# attention_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2), reuse=True)
num_layers = 2
encoding_embedding_size = 15
decoding_embedding_size = 15
learning_rate = 0.001

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       lr,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(source_letter_to_int),
                                                                       len(target_letter_to_int),
                                                                       encoding_embedding_size,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers)
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.predicted_ids, name='predictions')
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
        optimizer = tf.train.AdamOptimizer(lr)
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))
        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


# 将数据集分割为train和validation
train_source = source_int[batch_size:]
train_target = target_int[batch_size:]
# 留出一个batch进行验证
valid_source = source_int[:batch_size]
valid_target = target_int[:batch_size]
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(
    get_batches(valid_target, valid_source, batch_size,
                source_letter_to_int['<PAD>'],
                target_letter_to_int['<PAD>']))
display_step = 50  # 每隔50轮输出loss
checkpoint = "trained_model.ckpt"

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_i in range(1, epochs + 1):
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                get_batches(train_target, train_source, batch_size,
                            source_letter_to_int['<PAD>'],
                            target_letter_to_int['<PAD>'])):
            _, loss = sess.run(
                [train_op, cost],
                {input_data: sources_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths})

            if batch_i % display_step == 0:
                # 计算validation loss
                validation_loss = sess.run([cost],
                                           {input_data: valid_sources_batch,
                                            targets: valid_targets_batch,
                                            lr: learning_rate,
                                            target_sequence_length: valid_targets_lengths,
                                            source_sequence_length: valid_sources_lengths})
                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              loss,
                              validation_loss[0]))
    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')


def source_to_seq(text):
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int[
                                                                                                   '<PAD>']] * (
                                                                                                  sequence_length - len(
                                                                                                      text))


# 输入一个单词
input_word = 'common'
text = source_to_seq(input_word)

checkpoint = "./trained_model.ckpt"
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(input_word)] * batch_size,
                                      source_sequence_length: [len(input_word)] * batch_size})[0]
pad = source_letter_to_int["<PAD>"]
print('原始输入:', input_word)
print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))
print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))
