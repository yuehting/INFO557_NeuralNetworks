"""
The main code for the feedforward networks assignment.
See README.md for details.
"""
from typing import Tuple, Dict

import tensorflow
from tensorflow import keras
from keras import layers

def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """

    deep_model = keras.Sequential(
        [
            layers.Dense(32, input_shape=(n_inputs,), activation='linear'),
            layers.Dense(8, activation='linear'),
            layers.Dense(4, activation='linear'),
            layers.Dense(n_outputs, activation='linear')
        ]
    )
    wild_model = keras.Sequential(
        [
            layers.Dense(32, input_shape=(n_inputs,), activation='linear'),
            layers.Dense(8, activation='linear'),
            layers.Dense(n_outputs, activation='linear')
        ]
    )
    opt = keras.optimizers.Adam(learning_rate=0.01)
    deep_model.compile(optimizer=opt, loss='mse')
    wild_model.compile(optimizer=opt, loss='mse')

    return (deep_model, wild_model)

def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
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
    relu_model = keras.Sequential(
        [
            layers.Dense(8, input_dim=n_inputs, activation='relu'),
            layers.Dense(4, activation='relu'),
            layers.Dense(n_outputs, activation='sigmoid')
        ]
    )
    tanh_model = keras.Sequential(
        [
            layers.Dense(8, input_dim=n_inputs, activation='tanh'),
            layers.Dense(4, activation='tanh'),
            layers.Dense(n_outputs, activation='sigmoid')
        ]
    )

    opt = keras.optimizers.Adam(learning_rate=0.01)
    relu_model.compile(optimizer=opt, loss='binary_crossentropy')
    tanh_model.compile(optimizer=opt, loss='binary_crossentropy')
    return (relu_model, tanh_model)

def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
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
    nondropout_model = keras.Sequential(
        [
            layers.Dense(256, input_shape=(n_inputs,), activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_outputs, activation='softmax')
        ]
    )
    dropout_model = keras.Sequential(
        [
            layers.Dropout(0.2, input_shape=(n_inputs,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(n_outputs, activation='softmax')
        ]
    )

    opt = keras.optimizers.Adam(learning_rate=1e-3)
    nondropout_model.compile(optimizer=opt, loss='categorical_hinge')
    dropout_model.compile(optimizer=opt, loss='categorical_hinge')
    return (dropout_model, nondropout_model)

def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                Dict,
                                                tensorflow.keras.models.Model,
                                                Dict]:
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

    earlystopping =  {'callbacks': keras.callbacks.EarlyStopping(monitor='val_loss', mode='min')}
    nonearlystopping= {"callbacks": ''}

    earlystopping_model = keras.Sequential(
        [
            layers.Dense(32, input_shape=(n_inputs,), activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(n_outputs, activation='sigmoid')
        ]
    )
    nonearlystopping_model = keras.Sequential(
        [
            layers.Dense(32, input_shape=(n_inputs,), activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(n_outputs, activation='sigmoid')
        ]
    )

    opt = keras.optimizers.Adam(learning_rate=1e-3)
    earlystopping_model.compile(optimizer=opt, loss='binary_crossentropy')
    nonearlystopping_model.compile(optimizer=opt, loss='binary_crossentropy')
    return(earlystopping_model, earlystopping,
           nonearlystopping_model, nonearlystopping)
