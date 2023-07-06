#!/usr/bin/env python3
"""builing a neural network
with the Keras library"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """using Keras library"""
    model = K.Sequential()
    L = len(layers)

    for i in range(L):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_shape=(nx,),
                                     activation=activations[i],
                                     kernel_regularizer=K.regularizers.l2(
                                        lambtha)))
        else:
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=K.regularizers.l2(
                                        lambtha)))
        if i < L - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
