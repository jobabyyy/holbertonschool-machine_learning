#!/usr/bin/env python3
"""Function that builds a
transition layer"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer
    Returns the output of the transition layer and
    the number of filters within the output"""
    nb_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(filters=nb_filters, kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(X)

    return X, nb_filters
