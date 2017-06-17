#!/usr/bin/env python
"""
Trains the RNN with Meshuggah data.
"""
import argparse
import logging
import numpy as np
import os
import time

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.optimizers import Adam

import matplotlib.pyplot as plt


NORM_CQT_FRAMES = "../data/norm_cqt_frames.npy"
BATCH_SIZE = 64
N_FRAMES = 100  # Around 10 seconds of audio (HOP_SIZE * N_FRAMES / SR)
LSTM_OUT = 512
N_EPOCHS = 100
PATIENCE = 4
TEST = 0.1


def get_dataset(n_frames):
    """Split dataset into test and train."""
    # Read frames
    frames = np.load(NORM_CQT_FRAMES)
    F = frames.shape[-1]
    N = len(frames)

    # Test set
    n_test = int(N * TEST)
    n_test = n_test - n_test % n_frames
    x_test = frames[:n_test].reshape((-1, n_frames, F))
    y_test = frames[1:n_test + 1][n_frames - 1::n_frames]
    assert np.array_equal(y_test[0], x_test[1, 0])
    assert np.array_equal(y_test[1], x_test[2, 0])
    assert x_test.shape[0] == y_test.shape[0]

    # Train set
    n_train = N - n_test
    n_train = n_train - n_train % n_frames
    x_train = frames[n_test:n_test + n_train].reshape((-1, n_frames, F))
    y_train = frames[n_test + 1:n_test + 1 + n_train][n_frames - 1::n_frames]
    assert np.array_equal(y_train[0], x_train[1, 0])
    assert np.array_equal(y_train[1], x_train[2, 0])
    assert x_train.shape[0] == y_train.shape[0]

    return x_train, y_train, x_test, y_test


def sample_epoch(N, n_frames):
    frames = np.load(NORM_CQT_FRAMES)
    M = frames.shape[0]  # Number of frames in datset
    F = frames.shape[1]  # Number of CQT bins per frame

    # Get epoch
    x = np.zeros((N, n_frames, F))
    y = np.zeros((N, F))
    for i, n in enumerate(range(N)):
        start_i = np.random.random_integers(0, M - n_frames - 1)
        x[i] = frames[start_i:start_i + n_frames]
        y[i] = frames[start_i + n_frames]

    # Split into train and test
    n_test = int(N * TEST)
    x_test = x[:n_test]
    y_test = y[:n_test]
    x_train = x[n_test:]
    y_train = y[n_test:]

    return x_train, y_train, x_test, y_test


def process():
    x_train, y_train, x_test, y_test = get_dataset(N_FRAMES)
    cqt_bins = x_train.shape[-1]

    logging.info('Building model...')
    model = Sequential()
    # model.add(LSTM(LSTM_OUT, dropout=0.2, recurrent_dropout=0.2,
                   # input_shape=(N_FRAMES, cqt_bins)))
    # model.add(LSTM(LSTM_OUT, input_shape=(N_FRAMES, cqt_bins), return_sequences=True))
    # model.add(LSTM(LSTM_OUT))
    # model.add(Dense(cqt_bins, activation='linear'))

    model.add(LSTM(LSTM_OUT, input_shape=(N_FRAMES, cqt_bins)))
    model.add(Dense(cqt_bins, activation='linear'))

    # model.add(LSTM(cqt_bins, input_shape=(N_FRAMES, cqt_bins)))

    adam = Adam(lr=0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
                  metrics=['accuracy'])

    stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=PATIENCE, verbose=0, mode='auto')

    logging.info('Train...')
    N = 20000
    for i in range(N_EPOCHS):
        logging.info('Sampling data for epoch {}...'.format(1 + i))
        x_train, y_train, x_test, y_test = sample_epoch(N, N_FRAMES)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=1, validation_data=(x_test, y_test))

    # model.fit(x_train, y_train,
              # batch_size=BATCH_SIZE,
              # epochs=N_EPOCHS,
              # validation_data=(x_test, y_test),
              # callbacks=[stopping])
    score, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    logging.info('Test loss: {}'.format(score))
    logging.info('Test accuracy: {}'.format(acc))

    logging.info("Saving model...")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")


if __name__ == "__main__":
    start_time = time.time()

    # Setup the logger
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                        level=logging.INFO)
    process()
    logging.info("Done! Took %.2f seconds." % (time.time() - start_time))
