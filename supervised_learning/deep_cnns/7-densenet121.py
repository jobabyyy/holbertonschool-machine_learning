#!/usr/bin/env python3
"""Function that builds the DenseNet-121
and defines the dense block and
transition layers using dense block"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture
    and returns Keras Model"""

    X_inputs = K.Input(shape=(224, 224, 3))

    # convulational block inti
    X = K.layers.BatchNormalization()(X_inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(2 * growth_rate, (7, 7), strides=(2, 2),
                        padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                              padding='same')(X)

    # 1st dense block
    X, nb_filters = dense_block(X, 2 * growth_rate, growth_rate, 6)

    # 1st transition layer
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # 2nd dense block
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)

    # 2nd transition layer
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # dense layer 
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)

    # 3rd transition layer
    X, nb_filters = transition_layer(X, nb_filters, compression)

    # dense block
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    # Pooling
    X = K.layers.AveragePooling2D((7, 7))(X)
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer='he_normal')(X)

    # Keras Model instance
    model = K.Model(X_inputs=X_inputs, outputs=X)

    return model
