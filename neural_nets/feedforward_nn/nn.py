from typing import Tuple, Dict

from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.layers import Layer, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint


def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """
    deep = Sequential()
    deep.add(Dense(32,input_dim=n_inputs , activation='relu'))
    deep.add(Dense(16, activation='relu'))
    deep.add(Dense(8, activation='relu'))
    deep.add(Dense(16, activation='relu'))
    deep.add(Dense(32, activation='relu'))
    deep.add(Dense(8, activation='relu'))
    deep.add(Dense(16, activation='relu'))
    deep.add(Dense(32, activation='relu'))
    deep.add(Dense(n_outputs, activation='linear'))
    deep.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                 metrics=['accuracy'])

    wide = Sequential()
    wide.add(Dense(16,input_dim=n_inputs , activation='relu'))
    wide.add(Dense(8, activation='relu'))
    wide.add(Dense(256, activation='relu'))
    wide.add(Dense(n_outputs, activation='linear'))
    wide.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                 metrics=['accuracy'])

    return deep,wide 



def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """

    relu = simple_3layer_feedfrward(n_inputs,n_outputs,hidden_activations='relu',output_activiation='sigmoid',optimizer='rmsprop',loss='binary_crossentropy')
    tanh = simple_3layer_feedfrward(n_inputs,n_outputs,hidden_activations='tanh',output_activiation='sigmoid',optimizer='rmsprop',loss='binary_crossentropy')


    return relu, tanh


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """

    drop = simple_3layer_feedfrward(n_inputs,n_outputs,hidden_activations='relu',output_activiation='softmax',optimizer='adam',loss='categorical_crossentropy',dropout = .3)
    no_drop = simple_3layer_feedfrward(n_inputs,n_outputs,hidden_activations='relu',output_activiation='softmax',optimizer='adam',loss='categorical_crossentropy',dropout = 0)
    
    return drop , no_drop
    



def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[Model, Dict, Model, Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """

    model= simple_3layer_feedfrward(n_inputs,n_outputs,hidden_activations='relu',output_activiation='sigmoid',optimizer='adam',loss='binary_crossentropy')

    es = EarlyStopping(monitor='val_loss',  patience=5) # Findes early stopping using for the loss on the validation dataset, adding a delay of 5 to the trigger in terms of the number of epochs on which we would like to see no improvement. 
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss',save_best_only=True) #The callback will save the model to file, the one with best overall performance.
    early_fit_kwargs = dict(epochs=20, # Number of epochs
                    callbacks=[es,mc], # Early stopping
                    verbose=0, # Print description after each epoch
                    batch_size=128)
    
    late_fit_kwargs = dict(epochs=20, # Number of epochs
                verbose=0, # Print description after each epoch
                batch_size=128)
    
    return model,early_fit_kwargs,model,late_fit_kwargs



def simple_3layer_feedfrward(n_inputs,n_outputs,hidden_activations='relu',output_activiation='softmax',optimizer='adam',loss='categorical_crossentropy',dropout = 0):
    """ This function creates a simple 3layer feed forward network, with hidden layers size 128,256,128
    Given the input size, output size, hidden layers activation functions (one for all them), out put activatioin function, the optimizer, the loss function, and the dropout rate
    """
    
    model = Sequential()
    model.add(Dense(128,input_dim=n_inputs , activation=hidden_activations))
    if dropout: model.add(Dropout(dropout))
    model.add(Dense(256, activation=hidden_activations))
    if dropout: model.add(Dropout(dropout))
    model.add(Dense(128, activation=hidden_activations))
    if dropout: model.add(Dropout(dropout))
    model.add(Dense(n_outputs, activation= output_activiation))
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                 metrics=['accuracy'])
    return model

