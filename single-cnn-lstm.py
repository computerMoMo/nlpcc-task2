# -*- coding:utf-8 -*-
from __future__ import print_function
import sys
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
np.random.seed(1337)

if __name__ == '__main__':
    print("this is a single cnn-lstm model")
    # numpy文件路径
    data_path = 'exp-data/'
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]

    # CNN-LSTM参数设置
    lr_in = 0.7
    nb_epoch = int(sys.argv[3])
    nb_classes = 18
    max_len = 20
    word_dim = 300
    nb_filters = 250
    batch_size = 128
    filter_width = 3
    out_put_dim = 128

    # 读入训练和测试数据
    x_train = np.load(data_path + 'x-' + train_file_name + '.npy')
    y_train = np.load(data_path + 'y-' + train_file_name + '.npy')
    x_test = np.load(data_path + 'x-' + test_file_name + '.npy')
    y_test = np.load(data_path + 'y-' + test_file_name + '.npy')

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], word_dim, 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], word_dim, 1)
    y_train = np_utils.to_categorical(y_train, num_classes=nb_classes)
    y_test = np_utils.to_categorical(y_test, num_classes=nb_classes)

    print("x train data shape:", x_train.shape)
    print("x test data shape:", x_test.shape)
    print("y train data shape:", y_train.shape)
    print("y test data shape:", y_test.shape)

    # 建立模型
    model = Sequential()
    model.add(Conv2D(nb_filters, filter_width, word_dim, activation='relu', input_shape=(max_len, word_dim, 1),
                     name='cnn1'))
    model.add(MaxPooling2D(pool_size=(max_len - filter_width + 1, 1), name='maxpooling1'))
    model.add(Reshape((-1, nb_filters, 1)))
    model.add(LSTM(units=out_put_dim, input_shape=(nb_filters, 1), dropout=0.2, recurrent_dropout=0.2,
                   activation='relu', return_sequences=True, name='lstm_1'))
    model.add(Flatten())
    model.add(Dense(units=nb_classes, activation='softmax', name='softmax'))

    # 训练
    sgd = SGD(lr=lr_in, clipnorm=1.0)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(x_test, y_test),
              shuffle=True)

    # 测试集准确率
    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    print("program finished!")
