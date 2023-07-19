#!/usr/bin/env python3
"""Function that builds the ResNet-50
architecture"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture
    Returns the Keras model of ResNet-50"""
    input_shape = (224, 224, 3)

    X_input = K.Input(input_shape)

    # stage1
    X = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2),
               padding='same',
               kernel_initializer='he_normal')(X_input)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                     padding='same')(X)

    # stage2
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # stage3
    X = projection_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # stage4
    X = projection_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # stage5
    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # average pooling
    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                         padding='valid')(X)

    # output layer
    X = K.layers.Flatten()(X)
    X = K.layers.Dense(1000, activation='softmax')(X)

    # create model
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
