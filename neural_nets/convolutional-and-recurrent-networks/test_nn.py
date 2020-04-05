import os
import json
import random

import h5py
import keras
from keras import activations as A
from keras import losses as L
import numpy as np
import pytest
import tensorflow as tf

import nn


@pytest.fixture(autouse=True)
def set_seeds():
    # attempt to reduce randomness following Keras instructions:
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    seed_value = 42
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def test_toy_rnn(capsys):
    n_train = 20
    n_test = 10
    n_timesteps = 20
    n_features = 2

    # create random input for train and test
    train_in = np.random.randint(1, 11, (n_train, n_timesteps, n_features))
    test_in = np.random.randint(1, 11, (n_test, n_timesteps, n_features))

    # deterministically create output from the random input
    def out(matrix_in):
        matrix_out = np.zeros(shape=matrix_in.shape[:-1] + (1,))
        for i, example in enumerate(matrix_in):
            for j, [_, x1] in enumerate(example):
                [x0, _] = example[j - 3] if j >= 3 else [0., 0.]
                matrix_out[i, j] = x0 - x1
        return matrix_out
    train_out = out(train_in)
    test_out = out(test_in)

    # request a model
    input_shape = train_in.shape[1:]
    (_, _, n_outputs) = train_out.shape
    model, kwargs = nn.create_toy_rnn(input_shape, n_outputs)

    # check that model contains a recurrent layer
    assert any(isinstance(layer, keras.layers.RNN)
               for layer in base_layers(model))

    # check that model contains no convolutional layers
    assert all(not isinstance(layer, keras.layers.convolutional._Conv)
               for layer in base_layers(model))

    # check that output type and loss are appropriate
    assert all("mean" in l.__name__ for l in losses(model))
    assert output_activation(model) == A.linear

    # set training data, epochs and validation data
    kwargs.update(x=train_in, y=train_out,
                  epochs=20, validation_data=(test_in, test_out))

    # call fit, including any arguments supplied alongside the model
    model.fit(**kwargs)

    # make sure error is low enough
    rmse = root_mean_squared_error(model.predict(test_in), test_out)
    with capsys.disabled():
        print("\n{:.1f} RMSE for RNN on toy problem".format(rmse))
    assert rmse < 2


def test_image_cnn(capsys):
    # The data below was obtained as follows:
    # import numpy as np
    # import h5py
    # import keras.utils
    # from tensorflow.keras.datasets import mnist
    # (train_input, train_output), (test_input, test_output) = mnist.load_data()
    # train_input = np.expand_dims(train_input, axis=-1)
    # train_output = keras.utils.to_categorical(train_output).astype("?")
    # test_input = np.expand_dims(test_input, axis=-1)
    # test_output = keras.utils.to_categorical(test_output).astype("?")
    # with h5py.File('data/mnist.hdf5', 'w') as f:
    #     # include only every 100th training/testing example to limit dataset size
    #     train = f.create_group("train")
    #     train.create_dataset("input", compression="gzip", data=train_input[::100])
    #     train.create_dataset("output", compression="gzip", data=train_output[::100])
    #     test = f.create_group("test")
    #     test.create_dataset("input", compression="gzip", data=test_input[::100])
    #     test.create_dataset("output", compression="gzip", data=test_output[::100])

    with h5py.File("data/mnist.hdf5", 'r') as f:
        train = f["train"]
        train_out = np.array(train["output"])
        train_in = np.array(train["input"])
        test = f["test"]
        test_out = np.array(test["output"])
        test_in = np.array(test["input"])

    # request a model
    input_shape = train_in.shape[1:]
    (_, n_outputs) = train_out.shape
    model, kwargs = nn.create_mnist_cnn(input_shape, n_outputs)

    # check that model contains a convolutional layer
    assert any(isinstance(layer, keras.layers.convolutional._Conv)
               for layer in base_layers(model))

    # check that model contains no recurrent layers
    assert all(not isinstance(layer, keras.layers.RNN)
               for layer in base_layers(model))

    # check that output type and loss are appropriate
    assert all("categorical" in l.__name__ for l in losses(model))
    assert output_activation(model) == A.softmax

    # set training data, epochs and validation data
    kwargs.update(x=train_in, y=train_out,
                  epochs=10, validation_data=(test_in, test_out))

    # call fit, including any arguments supplied alongside the model
    model.fit(**kwargs)

    # make sure accuracy is high enough
    accuracy = multi_class_accuracy(model.predict(test_in), test_out)
    with capsys.disabled():
        print("\n{:.1%} accuracy for CNN on MNIST sample".format(accuracy))
    assert accuracy > 0.8


def test_text_rnn(capsys):
    # The data below was obtained as follows:
    # Download .csv files from
    #     https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection
    # import json
    # import re
    # import h5py
    # import numpy as np
    # import pandas as pd
    # names = ["1-Psy", "2-KatyPerry", "3-LMFAO","4-Eminem", "5-Shakira"]
    # dfs = [pd.read_csv("data/Youtube0{0}.csv".format(name)) for name in names]
    # tokenize = re.compile(r"\d+|[^\d\W]+|\S").findall
    # dfs_tokenized = [[tokenize(comment) for comment in df["CONTENT"]]
    #                  for df in dfs]
    #
    # index_to_token = [''] + sorted(set(token
    #                                    for comments in dfs_tokenized
    #                                    for tokens in comments
    #                                    for token in tokens))
    #
    # token_to_index = {c: i for i, c in enumerate(index_to_token)}
    #
    # max_tokens = max(len(tokens)
    #                  for comments in dfs_tokenized
    #                  for tokens in comments)
    #
    # with h5py.File('data/youtube-comments.hdf5', 'w') as f:
    #     f.attrs["vocabulary"] = json.dumps(index_to_token)
    #     for name, df, comments in zip(names, dfs, dfs_tokenized):
    #         matrix_in = np.zeros(shape=(len(comments), max_tokens))
    #         for i, tokens in enumerate(comments):
    #             for j, token in enumerate(tokens):
    #                 matrix_in[i, j] = token_to_index[token]
    #         matrix_out = df["CLASS"].values.reshape((-1, 1))
    #         group = f.create_group(name)
    #         group.create_dataset("input", compression="gzip", data=matrix_in)
    #         group.create_dataset("output", compression="gzip", data=matrix_out)
    with h5py.File("data/youtube-comments.hdf5", 'r') as f:
        vocabulary = json.loads(f.attrs["vocabulary"])
        train = f["1-Psy"]
        train_in = np.array(train["input"])[:, :200]
        train_out = np.array(train["output"])
        test = f["5-Shakira"]
        test_in = np.array(test["input"])[:, :200]
        test_out = np.array(test["output"])

    # request a model
    model, kwargs = nn.create_youtube_comment_rnn(vocabulary=vocabulary,
                                                  n_outputs=1)

    # check that model contains a recurrent layer
    assert any(isinstance(layer, keras.layers.RNN)
               for layer in base_layers(model))

    # check that model contains no convolutional layers
    assert all(not isinstance(layer, keras.layers.convolutional._Conv)
               for layer in base_layers(model))

    # check that output type and loss are appropriate
    assert all(any(x in l.__name__ for x in ["hinge", "crossentropy"])
               for l in losses(model))
    assert output_activation(model) == A.sigmoid

    # set training data, epochs and validation data
    kwargs.update(x=train_in, y=train_out,
                  epochs=10, validation_data=(test_in, test_out))

    # call fit, including any arguments supplied alongside the model
    model.fit(**kwargs)

    # make sure accuracy is high enough
    accuracy = binary_accuracy(model.predict(test_in), test_out)
    with capsys.disabled():
        print("\n{:.1%} accuracy for RNN on Youtube comments".format(accuracy))
    assert accuracy > 0.8


def test_text_cnn(capsys):
    # The data below was obtained as in test_text_rnn
    with h5py.File("data/youtube-comments.hdf5", 'r') as f:
        vocabulary = json.loads(f.attrs["vocabulary"])
        train = f["1-Psy"]
        train_in = np.array(train["input"])[:, :200]
        train_out = np.array(train["output"])
        test = f["5-Shakira"]
        test_in = np.array(test["input"])[:, :200]
        test_out = np.array(test["output"])

    # request a model
    model, kwargs = nn.create_youtube_comment_cnn(vocabulary=vocabulary,
                                                  n_outputs=1)

    # check that model contains a convolutional layer
    assert any(isinstance(layer, keras.layers.convolutional._Conv)
               for layer in base_layers(model))

    # check that model contains no recurrent layers
    assert all(not isinstance(layer, keras.layers.RNN)
               for layer in base_layers(model))

    # check that output type and loss are appropriate
    assert all(any(x in l.__name__ for x in ["hinge", "crossentropy"])
               for l in losses(model))
    assert output_activation(model) == A.sigmoid

    # set training data, epochs and validation data
    kwargs.update(x=train_in, y=train_out,
                  epochs=10, validation_data=(test_in, test_out))

    # call fit, including any arguments supplied alongside the model
    model.fit(**kwargs)

    # make sure accuracy is high enough
    accuracy = binary_accuracy(model.predict(test_in), test_out)
    with capsys.disabled():
        print("\n{:.1%} accuracy for CNN on Youtube comments".format(accuracy))
    assert accuracy > 0.8


def losses(model):
    return [loss.fn if isinstance(loss, L.LossFunctionWrapper) else loss
            for loss in model.loss_functions]


def output_activation(model):
    return model.layers[-1].activation


def base_layers(model):
    for layer in model.layers:
        while isinstance(layer, keras.layers.wrappers.Wrapper):
            layer = layer.layer
        yield layer


def root_mean_squared_error(system: np.ndarray, human: np.ndarray):
    return ((system - human) ** 2).mean() ** 0.5


def multi_class_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.argmax(system, axis=1) == np.argmax(human, axis=1))


def binary_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.round(system) == human)
