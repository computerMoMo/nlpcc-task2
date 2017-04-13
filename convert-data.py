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
def read_text_data(tag_file, text_file_path, cv=10, clean_string=True, cut_words=False):
    revs = []
    vocab = defaultdict(float)

    # read tag file
    tag_dict = dict()
    try:
        tag_reader = open(tag_file, 'r')
        tag_val = 0
        for tag_line in tag_reader.readlines():
            tag_list = tag_line.split('\t')
            tag_key = tag_list[0]
            tag_dict.update({tag_key: tag_val})
            tag_val += 1
    except:
        print("error in opening ", tag_file)

    print ("tag dict: ", tag_dict)

    # read lines in text
    try:
        with open(text_file_path, 'r') as file_reader:
            while True:
                str_line = file_reader.readline()
                if not str_line:
                    break
                str_line = str_line.strip()
                str_list = str_line.split('\t')
                line_tag = str_list[0]

                orig_str = "".join(str_list[1:])
                if clean_string:
                    orig_str = clean_str(orig_str).lower()
                else:
                    orig_str = orig_str.lower()

                # cut words
                if cut_words:
                    words_list = jieba.lcut(orig_str, cut_all=True)
                else:
                    words_list = orig_str.split()
                words_set = set(words_list)
                for word in words_set:
                    vocab[word] += 1
                datum = {
                    'flag': tag_dict.get(line_tag),
                    'word_list': words_list,
                    'split': np.random.randint(0, cv)
                }
                revs.append(datum)
    except:
        print("errors in opening ", text_file_path)

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
    text_data_file = sys.argv[1]
    tag_file = sys.argv[2]
    revs, vocab = read_text_data(tag_file, text_data_file, cut_words=False)

    # generate numpy
    numpy_save_path = sys.argv[3]
    numpy_save_name = sys.argv[4]
    print ("generate numpy")
    sent_max_len = 20
    word_dim = 300
    x_array, y_array = words_convert(revs, max_len=sent_max_len, word_dim=word_dim)
    print("x array shape:", x_array.shape)
    print("y array shape:", y_array.shape)
    np.save(numpy_save_path + 'x-' + numpy_save_name + '.npy', x_array)
    np.save(numpy_save_path + 'y-' + numpy_save_name + '.npy', y_array)
    print("program finished!")
