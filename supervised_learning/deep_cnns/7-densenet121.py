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

    inputs = K.Input(shape=(224, 224, 3))

    # convulational block inti
    x = K.layers.BatchNormalization()(inputs)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(2 * growth_rate, (7, 7), strides=(2, 2),
                        padding='same',
                        kernel_initializer='he_normal')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                              padding='same')(x)

    # 1st dense block
    x, nb_filters = dense_block(x, 2 * growth_rate, growth_rate, 6)

    # 1st transition layer
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # 2nd dense block
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)

    # 2nd transition layer
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # dense layer
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)

    # 3rd transition layer
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # dense block
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    # Pooling
    x = K.layers.AveragePooling2D((7, 7))(x)
    x = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer='he_normal')(x)

    # Keras Model instance
    model = K.Model(inputs=inputs, outputs=x)

    return model
