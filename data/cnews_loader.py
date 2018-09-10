# -*- coding: utf-8 -*-

import numpy as np
import os

base_dir = 'data'
stopwords_dir = os.path.join(base_dir, 'stop_words.txt')


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def process_source_file(file_dir, letter_to_id):
    letter_ids = []
    len_ = []
    with open_file(file_dir) as f:
        for line in f:
            letter_id = []
            conts = list(line.strip())
            for con in conts:
                letter_id.append(letter_to_id.get(con, letter_to_id['<UNK>']))
            letter_ids.append(letter_id)
            len_.append(len(letter_id))
    # return pad_sequences(letter_ids, maxlen=source_seq_length, value=letter_to_id['<PAD>']), np.array(len_)
    return np.array(letter_ids), np.array(len_)


def process_target_file(file_dir, letter_to_id):
    letter_ids = []
    len_ = []
    with open_file(file_dir) as f:
        for line in f:
            letter_id = []
            conts = list(line.strip())
            for con in conts:
                letter_id.append(letter_to_id.get(con, letter_to_id['<UNK>']))
            letter_id.append(letter_to_id['<EOS>'])
            letter_ids.append(letter_id)
            # len_.append(target_seq_length if len(letter_id) > target_seq_length else len(letter_id))
            len_.append(len(letter_id))
    # padded = pad_sequences(letter_ids, maxlen=target_seq_length, value=letter_to_id['<PAD>'])
    # cut掉最后一个字符，然后最前面加上<GO>字符
    # go_ = np.array([[letter_to_id['<GO>']] * len(letter_ids)]).transpose()
    # cuted_ = np.array(padded)[:, 0:len(padded[0])-1]
    # return np.hstack((go_, cuted_)), np.array(len_)
    return np.array(letter_ids), np.array(len_)


def process_predict_input(query, letter_to_id):
    letter_id = []
    conts = list(query.strip())
    for con in conts:
        letter_id.append(letter_to_id.get(con, letter_to_id['<UNK>']))
    letter_ids = [letter_id]
    len_ = [len(letter_id)]
    # return pad_sequences(letter_ids, maxlen=7, value=letter_to_id['<PAD>']), np.array(len_)
    return letter_ids, len_


def build_vocab(total_dir, vocab_dir):
    print("building vacab...")
    final_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    with open_file(total_dir) as f:
        for line in f:
            chars = list(line.strip())
            for char in chars:
                final_words.append(char)
    open_file(vocab_dir, mode='w').write('\n'.join(set(final_words)) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    return word_to_id, id_to_word


def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def clip_batch(sources_batch, targets_batch, source_pad_int, target_pad_int):
    pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
    pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
    # 记录每条记录的长度
    targets_lengths = []
    for target in targets_batch:
        targets_lengths.append(len(target))
    source_lengths = []
    for source in sources_batch:
        source_lengths.append(len(source))
    return pad_sources_batch, source_lengths, pad_targets_batch, targets_lengths


# def batch_iter(source_train, len_source_train, target_train, len_target_train, batch_size, source_pad_int, target_pad_int):
def batch_iter(source_train, target_train, batch_size, source_pad_int, target_pad_int):
    """生成批次数据"""
    data_len = len(source_train)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    source_train_shuffle = []
    # len_source_train_shuffle = []
    target_train_shuffle = []
    # len_target_train_shuffle = []
    for i in range(len(indices)):
        source_train_shuffle.append(source_train[indices[i]])
        # len_source_train_shuffle.append(len_source_train[indices[i]])
        target_train_shuffle.append(target_train[indices[i]])
        # len_target_train_shuffle.append(len_target_train[indices[i]])

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        # yield source_train_shuffle[start_id:end_id], len_source_train_shuffle[start_id:end_id], target_train_shuffle[start_id:end_id], len_target_train_shuffle[start_id:end_id]
        yield clip_batch(source_train_shuffle[start_id:end_id], target_train_shuffle[start_id:end_id], source_pad_int, target_pad_int)
