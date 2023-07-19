#!/usr/bin/env python3
"""Function that builds the inception
network"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """model consisting of multiple inception
    blocks. It is designed for image classification.
    func is to input data of shape given below and
    is to return inception Neural Network model. """
    # Input shape
    X_input = K.Input(shape=(224, 224, 3))

    # input layer
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        padding='same', activation='relu')(X_input)
    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)

    # conv layer num1
    X = K.layers.Conv2D(192, (3, 3), activation='relu', padding='same')(X)
    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)

    # conv layer num2
    X = inception_block(X, [64, 96, 128, 16, 32, 32])
    X = inception_block(X, [128, 128, 192, 32, 96, 64])
    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)

    # inception blocks designed for image classification
    X = inception_block(X, [192, 96, 208, 16, 48, 64])
    X = inception_block(X, [160, 112, 224, 24, 64, 64])
    X = inception_block(X, [128, 128, 256, 24, 64, 64])
    X = inception_block(X, [112, 144, 288, 32, 64, 64])
    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(X)

    # inception 5a and 5b
    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = inception_block(X, [384, 192, 384, 48, 128, 128])

    # final Layers
    X = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=(7, 7),
                                  padding='valid')(X)
    X = K.layers.Dropout(0.4)(X)  # dropout
    X = K.layers.Dense(1000, activation='softmax')(X)

    # create the model
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
