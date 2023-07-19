#!/usr/bin/env python3
"""Func that defines the identity
block which will consist of
3 cnn layers w batch normalization
and ReLu activation."""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Builds an identity block.
    Contains multiple layers including output
    from previous layer.
    3 filters added."""
    F11, F3, F12 = filters

    X_shortcut = A_prev

    # beginning main path
    X = K.layers.Conv2D(F11, kernel_size=(1, 1), strides=(1, 1),
                        padding='valid',
                        kernel_initializer='he_normal')(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # adding 2nd component to main
    X = K.layers.Conv2D(F3, kernel_size=(3, 3), strides=(1, 1),
                        padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # 3rd component of main path added
    X = K.layers.Conv2D(F12, kernel_size=(1, 1), strides=(1, 1),
                        padding='valid',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add the input value to the main path
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
