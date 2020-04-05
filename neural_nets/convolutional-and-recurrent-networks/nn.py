from typing import Tuple, List, Dict

import keras

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, SimpleRNN, Bidirectional
from keras.layers import Conv2D, Conv1D, MaxPooling2D, GlobalMaxPooling1D, Flatten

def create_toy_rnn(input_shape: tuple,
                   n_outputs: int) -> Tuple[keras.Model, Dict]:
    """Creates a recurrent neural network for a toy problem.

    The network will take as input a sequence of number pairs, (x_{t}, y_{t}),
    where t is the time step. It must learn to produce x_{t-3} - y{t} as the
    output of time step t.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = Sequential()
    model.add(SimpleRNN(256, input_shape=input_shape, return_sequences=True))
    model.add(Dense(n_outputs, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    kwargs = dict(epochs=20, # Number of epochs
                    verbose=0, # Print description after each epoch
                 )
    
    return model,kwargs


def create_mnist_cnn(input_shape: tuple,
                     n_outputs: int) -> Tuple[keras.Model, Dict]:
    """Creates a convolutional neural network for digit classification.

    The network will take as input a 28x28 grayscale image, and produce as
    output one of the digits 0 through 9. The network will be trained and tested
    on a fraction of the MNIST data: http://yann.lecun.com/exdb/mnist/

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """

    model=Sequential()
    model.add(Conv2D(64,(5,5),input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(n_outputs, activation = 'softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    kwargs = dict(epochs=20, # Number of epochs
                    verbose=0, # Print description after each epoch
                 )
    
    return model , kwargs
    


def create_youtube_comment_rnn(vocabulary: List[str],
                               n_outputs: int) -> Tuple[keras.Model, Dict]:
    """Creates a recurrent neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model = Sequential()
    
    model.add(
        Embedding(input_dim=len(vocabulary),
                  output_dim=64))
    model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)))
    model.add(Dense(n_outputs, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    kwargs = dict(epochs=20, # Number of epochs
                    verbose=0, # Print description after each epoch
                 )
    
    return model,kwargs
    


def create_youtube_comment_cnn(vocabulary: List[str],
                               n_outputs: int) -> Tuple[keras.Model, Dict]:
    """Creates a convolutional neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    model=Sequential()
    model.add(
        Embedding(input_dim=len(vocabulary),
                  output_dim=64))
    
    model.add(Conv1D(64,5))
    model.add(Conv1D(32,5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    kwargs = dict(epochs=20, # Number of epochs
                    verbose=0, # Print description after each epoch
                 )
    return model, kwargs
