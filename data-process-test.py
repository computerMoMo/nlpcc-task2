# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import os
import random
import jieba
np.random.seed(1337)


def build_data_cv(tag_list, data_path, data_str, cv=10):
    """
    Loads data and split into 10 folds.
    """
    revs = []

    vocab = defaultdict(float)

    for tag in tag_list:
        text_file = data_path+data_str+'/'+data_str+'-'+tag+'.txt'
        flag = 0
        with open(text_file, "rb") as f:
            for line in f:
                rev = []
                rev.append(line.strip())
                orig_rev = " ".join(rev)
                words = set(orig_rev.split())
                for word in words:
                    vocab[word] += 1
                datum = {"y": flag,
                         "text": orig_rev,
                         "num_words": len(orig_rev.split()),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)
        flag += 1
    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)

def get_idx_from_sent(sent, word_idx_map, word_vector, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_vector[word_idx_map[word]])
            if len(x) == max_l:
                break
    while len(x) < max_l:
        x.append(word_vector[0])
    return x

def make_idx_data_cv(revs, word_idx_map, word_vector, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    x_target = []
    y_target = []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, word_vector, max_l, k, filter_h)
        x_target.append(sent)
        y_target.append(rev["y"])
    x_target_array = np.asarray(x_target, dtype="float32")
    y_target_array = np.asarray(y_target, dtype='uint8')

    return x_target_array, y_target_array

if __name__ == "__main__":
    w2v_file = '/home/zmh/exp-data/word-vectors/baike_vector.bin'
    data_path = 'exp-data/'
    data_str = sys.argv[1]
    # load data
    print('load data')
    tag_list = ['history', 'military', 'baby', 'world', 'tech', 'game','society',
                'sports', 'travel', 'car', 'food', 'entertainment', 'finance', 'fashion',
                'discovery', 'story', 'regimen', 'essay']
    revs, vocab = build_data_cv(tag_list, data_path, data_str, cv=10)
    print('data loaded')
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print("number of sentences: " + str(len(revs)))
    print("vocab size: " + str(len(vocab)))
    print("max sentence length: " + str(max_l))
    # load w2v
    w2v = load_bin_vec(w2v_file, vocab)
    print("word2vec loaded!")
    print("num words already in word2vec: " + str(len(w2v)))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    # generate numpy file
    print("generate numpy file...")
    x_train, y_train = make_idx_data_cv(revs, word_idx_map, word_vector=W, max_l=20, k=300, filter_h=5)
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    np.save(data_path + 'x-data-'+data_str+'-ex.npy', x_train)
    np.save(data_path + 'y-data-'+data_str+'-ex.npy', y_train)
