#!/usr/bin/env python3
"""Function that builds the inceptiom
network"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """model consisting of multiple inception
    blocks. It is designed for image classification.
    func is to input data of shape given below and
    is to return inception Neural Network model. """
    input_shape = (224, 224, 3)  # input data

    # input layer
    inputs = K.Input(shape=input_shape)

    # conv layer num1
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        activation='relu')(inputs)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # conv layer num2
    x = K.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = K.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # inception blocks designed for image classification
    x = inception_block(x, [64, 96, 128, 16, 32, 32])  # inception 3a
    x = inception_block(x, [128, 128, 192, 32, 96, 64])  # inception 3b

    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])  # inception 4a
    x = inception_block(x, [160, 112, 224, 24, 64, 64])  # inception 4b
    x = inception_block(x, [128, 128, 256, 24, 64, 64])  # inception 4c
    x = inception_block(x, [112, 144, 288, 32, 64, 64])  # inception 4d
    x = inception_block(x, [256, 160, 320, 32, 128, 128])  # inception 4e

    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])  # inception 5a
    x = inception_block(x, [384, 192, 384, 48, 128, 128])  # inception 5b

    # final layers
    x = K.layers.AveragePooling2D((7, 7), strides=(1, 1))(x)  # pooling
    x = K.layers.Flatten()(x)  # flatten
    x = K.layers.Dropout(0.4)(x)  # dropout
    outputs = K.layers.Dense(1000, activation='softmax')(x)

    # create the model
    model = K.models.Model(inputs, outputs)
    return model
