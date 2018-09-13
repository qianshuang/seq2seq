# -*- coding: utf-8 -*-

import sys

# from basic_model import *
# from attention_model import *
# from bilstm_attention_model import *
# from beamsearch_bilstm_attention_model import *
# from beamsearch_basic_model import *
from beamsearch_bilstm_attention_model import *
from data.cnews_loader import *

import time
from datetime import timedelta


base_dir = 'data'
source_train_dir = os.path.join(base_dir, 'letters_source_train.txt')
source_test_dir = os.path.join(base_dir, 'letters_source_test.txt')
target_train_dir = os.path.join(base_dir, 'letters_target_train.txt')
target_test_dir = os.path.join(base_dir, 'letters_target_test.txt')
source_vocab_dir = os.path.join(base_dir, 'letters_source.vocab.txt')
target_vocab_dir = os.path.join(base_dir, 'letters_target.vocab.txt')


save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(sess, source_test, target_test):
    """评估在某一数据上的准确率和损失"""
    data_len = len(source_test)
    batch_eval = batch_iter(source_test, target_test, config.batch_size, source_letter_to_id['<PAD>'], target_letter_to_id['<PAD>'])
    total_loss = 0.
    for source_train_batch, len_source_train_batch, target_train_batch, len_target_train_batch in batch_eval:
        batch_len = len(source_train_batch)
        feed_dict = {
            model.source: source_train_batch,
            model.target: target_train_batch,
            model.source_sequence_length: len_source_train_batch,
            model.target_sequence_length: len_target_train_batch,
            # model.is_train: True
        }
        loss = sess.run([model.loss], feed_dict=feed_dict)
        loss = np.mean(loss)
        total_loss += loss * batch_len
    return total_loss / data_len


def train():
    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 载入训练集与验证集
    print("Loading data...")
    source_train, len_source_train = process_source_file(source_train_dir, source_letter_to_id)
    source_test, len_source_test = process_source_file(source_test_dir, source_letter_to_id)
    target_train, len_target_train = process_target_file(target_train_dir, target_letter_to_id)
    target_test, len_target_test = process_target_file(target_test_dir, target_letter_to_id)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0              # 总批次
    best_val_loss = sys.float_info.max           # 最佳验证集效果
    last_improved = 0            # 记录上一次提升批次
    require_improvement = 100   # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(source_train, target_train, config.batch_size, source_letter_to_id['<PAD>'], target_letter_to_id['<PAD>'])
        for source_train_batch, len_source_train_batch, target_train_batch, len_target_train_batch in batch_train:
            feed_dict = {
                model.source: source_train_batch,
                model.target: target_train_batch,
                model.source_sequence_length: len_source_train_batch,
                model.target_sequence_length: len_target_train_batch,
                # model.is_train: True
            }

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集上的性能
                loss_train = np.mean(session.run([model.loss], feed_dict=feed_dict))
                loss_val = evaluate(session, source_test, target_test)

                if loss_val < best_val_loss:
                    # 保存最好结果
                    best_val_loss = loss_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Val Loss: {2:>6.2}, Time: {3} {4}'
                print(msg.format(total_batch, loss_train, loss_val, time_dif, improved_str))

            session.run(model.train_op, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Testing...")
    input_ = 'common'
    print('原始输入: ', input_)
    source_test, len_source_test = process_predict_input(input_, source_letter_to_id)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    feed_dict = {
        model.source: source_test,
        model.source_sequence_length: len_source_test,
        # model.target_sequence_length: len_source_test,
        # model.is_train: False
    }
    result_ids = session.run(model.result_ids, feed_dict=feed_dict)[0]
    # print('输出: {}'.format("".join([target_id_to_letter[i] for i in result_ids])))
    # beam search输出
    for x in result_ids:
        res = []
        for i in x:
            if i in target_id_to_letter:
                res.append(target_id_to_letter[i])
        print("".join(res))


if __name__ == '__main__':
    print('Configuring model...')
    config = TCNNConfig()
    if not os.path.exists(source_vocab_dir):
        build_vocab(source_train_dir, source_vocab_dir)
    if not os.path.exists(target_vocab_dir):
        build_vocab(target_train_dir, target_vocab_dir)
    source_letter_to_id, source_id_to_letter = read_vocab(source_vocab_dir)
    target_letter_to_id, target_id_to_letter = read_vocab(target_vocab_dir)
    config.source_vocab_size = len(source_letter_to_id)
    config.target_vocab_size = len(target_letter_to_id)
    config.target_letter_to_id = target_letter_to_id

    model = TextCNN(config)

    train()
    test()
