#!/usr/bin/env python3
"""Function that defines the
projection block which consists
of cnn layers"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """builds a projection block.
    Returns the activated output of the projection block"""
    F11, F3, F12 = filters

    X_shortcut = A_prev

    """Main Path"""

    # main path
    X = K.layers.Conv2D(F11, kernel_size=(1, 1), strides=(s, s),
                        padding='valid',
                        kernel_initializer='he_normal')(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # 2nd component of main path
    X = K.layers.Conv2D(F3, kernel_size=(3, 3), strides=(1, 1),
                        padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # 3rd component of main path
    X = K.layers.Conv2D(F12, kernel_size=(1, 1), strides=(1, 1),
                        padding='valid',
                        kernel_initializer='he_normal')(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    """Shortcut Path"""

    # cnn layer
    X_shortcut = K.layers.Conv2D(F12, kernel_size=(1, 1), strides=(s, s),
                                 padding='valid',
                                 kernel_initializer='he_normal')(X_shortcut)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Add the shortcut value to the main path
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation('relu')(X)

    return X
