# -*- coding:utf-8 -*-
from __future__ import print_function
import sys
import re
import numpy as np
import gensim
import os
import jieba
from collections import defaultdict
from gensim.models import KeyedVectors

global word_vectors


# clean English strings
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


# read text data
def read_text_data(file_list, cv=10, clean_string=True, chinese=False):
    revs = []
    vocab = defaultdict(float)
    flag = 0
    # read lines in text
    for text_file in file_list:
        try:
            with open(text_file, 'r') as file_reader:
                while True:
                    str_line = file_reader.readline()
                    if not str_line:
                        break
                    str_line = str_line.strip()
                    if clean_string:
                        orig_str = clean_str(str_line).lower()
                    else:
                        orig_str = str_line.lower()

                    # cut words
                    if chinese:
                        words_list = jieba.lcut(orig_str, cut_all=True)
                    else:
                        words_list = orig_str.split()
                    words_set = set(words_list)

                    for word in words_set:
                        vocab[word] += 1
                    datum = {'flag': str(flag),
                             'word_list': words_list,
                             'num_words': len(words_list),
                             'split': np.random.randint(0, cv)}
                    revs.append(datum)
        except:
            print("errors in opening ", text_file)
        flag += 1

    return revs, vocab


# read word vectors
def read_word_vectors(word_list, max_len=100, word_dim=300):
    word_x_target = []
    for word in word_list:
        if word in word_vectors.vocab:
            word_x_target.append(word_vectors.word_vec(word))
        else:
            vector_random = np.random.uniform(-0.25, 0.25, word_dim)
            word_x_target.append(vector_random)
        if len(word_x_target) >= max_len:
            break
    zero_vector = np.zeros(shape=(word_dim,))
    while len(word_x_target)< max_len:
        word_x_target.append(zero_vector)
    return word_x_target


# convert words to numpy
def words_convert(revs, max_len=100, word_dim=300):
    x_target = []
    y_target = []
    for rev in revs:
        word_x_target = read_word_vectors(rev['word_list'], max_len, word_dim)
        x_target.append(word_x_target)
        y_target.append(rev['flag'])
    x_target = np.asarray(x_target, dtype=np.float32)
    y_target = np.asarray(y_target, dtype=np.uint8)
    return x_target, y_target


if __name__ == '__main__':
    # load word vectors
    print("loading word vectors")
    word_vectors_path = '/home/zmh/exp-data/word-vectors/baike-vector.bin'
    word_vectors = KeyedVectors.load_word2vec_format(word_vectors_path, binary=True)

    # open text files
    print ("read text data")
    text_data_file_path = sys.argv[1]
    text_file_list = os.listdir(text_data_file_path)
    revs, vocab = read_text_data(text_file_list, chinese=True)

    # generate numpy
    numpy_save_path = sys.argv[2]
    numpy_save_name = sys.argv[3]
    print ("generate numpy")
    sent_max_len = 50
    word_dim = 300
    x_array, y_array = words_convert(revs, max_len=sent_max_len, word_dim=word_dim)
    print("x array shape:", x_array.shape)
    print("y array shape:", y_array.shape)
    np.save(numpy_save_path + 'x-' + numpy_save_name + '.npy', x_array)
    np.save(numpy_save_path + 'y-' + numpy_save_name + '.npy', y_array)
    print("program finished!")
