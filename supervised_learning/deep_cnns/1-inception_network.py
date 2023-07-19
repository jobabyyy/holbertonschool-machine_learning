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
    # input shape
    input_shape = (224, 224, 3)

    # input layer
    inputs = K.Input(shape=input_shape)

    # conv layer num1
    x = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                        activation='relu')(inputs)
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(x)

    # conv layer num2
    x = K.layers.Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    x = K.layers.Conv2D(192, kernel_size=(3, 3), padding='same',
                        activation='relu')(x)
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(x)

    # inception block designed for image classification
    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])  # Inception 3b
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(x)

    x = inception_block(x, [192, 96, 208, 16, 48, 64])  # Inception 4a
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])  # Inception 4c
    x = inception_block(x, [112, 144, 288, 32, 64, 64])  # Inception 4d
    x = inception_block(x, [256, 160, 320, 32, 128, 128])  # Inception 4e
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(x)

    x = inception_block(x, [256, 160, 320, 32, 128, 128])  # Inception 5a
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    # final layers
    x = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    x = K.layers.Dropout(0.4)(x)
    x = K.layers.Dense(1000, activation='softmax')(x)

    # create layers
    model = K.models.Model(inputs=inputs, outputs=x)

    return model
