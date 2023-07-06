#!/usr/bin/env python3
"""builing a neural network
with the Keras library"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """still using Keras library"""
    x = K.Input(shape=(nx,))
    reg = K.regularizers.l2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            y = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=reg)(x)
        else:
            y = K.layers.Dense(layers[i], activation=activations[i],
                               kernel_regularizer=reg)(y)
        if i < len(layers) - 1:
            y = K.layers.Dropout(1 - keep_prob)(y)

    model = K.models.Model(inputs=x, outputs=y)

    return model
