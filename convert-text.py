# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import sys
import jieba
import re
import gensim
import numpy as np
from collections import defaultdict
from gensim.models import KeyedVectors

global word_vectors

# 英文字符串的清洗
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\']", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


# jieba分词
def cut_words(string):
    return " ".join(jieba.lcut(string))


# 读入文本和标签
def read_text_data(tag_file, text_file_name, cv=10, clean=False, cut=False):
    revs = []
    vocab = defaultdict(float)

    # 读入标签文件
    tag_dict = dict()
    try:
        with open(tag_file, 'r') as file_reader:
            tag_val = 0
            for tag_line in file_reader.readlines():
                tag_line = tag_line.strip('\r\n')
                tag_line = tag_line.strip('\n')
                tag_list = tag_line.split('\t')
                tag_key = tag_list[0]
                tag_dict.update({tag_key: tag_val})
                tag_val += 1
    except:
        print("errors in opening ", tag_file)

    # 读入文本文件
    # test_f_w = open('exp-data/test-out.txt', 'w')
    try:
        with open(text_file_name, 'r') as file_reader:
            while True:
                str_line = file_reader.readline().decode(encoding='utf-8', errors='ignore')
                if not str_line:
                    break
                str_list = str_line.split('\t')
                str_tag = str_list[0]
                origin_str = "".join(str_list[1:])
                # test_f_w.write(str(tag_dict.get(str_tag))+'\t'+origin_str)
                if clean:
                    origin_str = clean_str(origin_str)
                if cut:
                    origin_str = cut_words(origin_str)

                word_list = origin_str.split()
                for word in word_list:
                    vocab[word] += 1
                data_dic = {
                    'tag': tag_dict.get(str_tag),
                    'word_list': word_list,
                    'split': np.random.randint(0, cv)
                }
                revs.append(data_dic)
    except:
        print("errors in opening ", text_file_name)

    return revs, vocab


# 生成词向量中不存在的词的词向量
def add_unknown_words(text_vocabs, word_dim=300):
    vectors_vocab = word_vectors.vocab
    unknown_words_dic = dict()
    for word in text_vocabs:
        if word not in vectors_vocab:
            unknown_words_dic.update({word: np.random.uniform(-0.25, 0.25, word_dim)})
    return unknown_words_dic


# 将文本中的每一行文本表示成一个二维numpy数组
def convert_sent_data(word_list, unk_word_vectors, max_len=100, word_dim=300):
    sent_x_target = []
    vec_vocab = word_vectors.vocab
    for word in word_list:
        if word in vec_vocab:
            sent_x_target.append(word_vectors.word_vec(word))
        else:
            sent_x_target.append(unknown_words_vectors.get(word))
        if len(sent_x_target) >= max_len:
            break
    zero_vec = np.zeros(shape=(word_dim,))
    while len(sent_x_target) < max_len:
        sent_x_target.append(zero_vec)
    return sent_x_target


# 生成整个文本的表示向量，为一个三维numpy数组
def convert_text_data(revs, unk_word_vectors, max_len=100, word_dim=300):
    text_x_target = []
    text_y_target = []
    for rev in revs:
        sent_x_target = convert_sent_data(rev['word_list'], unk_word_vectors, max_len, word_dim)
        text_x_target.append(sent_x_target)
        text_y_target.append(rev['tag'])
    x_target_array = np.asarray(text_x_target, dtype=np.float32)
    y_target_array = np.asarray(text_y_target, dtype=np.uint8)
    return x_target_array, y_target_array


if __name__ == '__main__':
    # 读取文本并生成词典
    print("reading text data...")
    text_data_path = sys.argv[1]
    text_revs, text_vocab = read_text_data('exp-data/tag2id.txt', text_data_path, cv=10, clean=False, cut=False)

    # 读取词向量
    print("loading word vectors...")
    # word_vectors_path = '/home/zmh/exp-data/word-vectors/baike-vector.bin'
    word_vectors_path = sys.argv[4]
    word_vectors = KeyedVectors.load_word2vec_format(word_vectors_path, binary=True)

    # 生成numpy数组
    numpy_save_path = 'exp-data/'
    numpy_save_name = sys.argv[2]
    vectors_dim = 300
    max_sent_len = int(sys.argv[3])
    print("generate numpy arrays")
    print("vocabulary of word vector file size is :", len(word_vectors.vocab))
    print("vocabulary of text size is :", len(text_vocab))
    unknown_words_vectors = add_unknown_words(text_vocab, word_dim=vectors_dim)
    print("nums of words don't exist in vector file is:", len(unknown_words_vectors))
    x_array, y_array = convert_text_data(text_revs, unknown_words_vectors, max_len=max_sent_len, word_dim=vectors_dim)
    print("x array shape:", x_array.shape)
    print("y array shape:", y_array.shape)
    np.save(numpy_save_path + 'x-' + numpy_save_name + '.npy', x_array)
    np.save(numpy_save_path + 'y-' + numpy_save_name + '.npy', y_array)
    print("program finished!")
