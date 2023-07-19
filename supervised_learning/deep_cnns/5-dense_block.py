#!/usr/bin/env python3
"""Builds a dense block and
concatenates output of each layer with
block and num of filters"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """adding parameters"""
    concat_output = X

    for i in range(layers):
        # Bottleneck layer
        X = K.layers.Conv2D(4 * growth_rate,
                            (1, 1),
                            kernel_initializer='he_normal')(X)
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation('relu')(X)

        # Convolutional layer
        X = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer='he_normal')(X)
        X = K.layers.BatchNormalization()(X)
        X = K.layers.Activation('relu')(X)

        # Concatenate output
        concat_output = K.layers.Concatenate()([concat_output, X])
        nb_filters += growth_rate

    return concat_output, nb_filters
