#!/usr/bin/env python3
"""func that builds a modified version
of the LeNet-5 architecture
using Keras"""


import tensorflow.keras as K


def lenet5(X):
    """building a modified version
    of LeNet-5"""
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            activation='relu',
                            kernel_initializer='he_normal')(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    # Convolutional Layer 2
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                            activation='relu',
                            kernel_initializer='he_normal')(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Flatten
    flatten = K.layers.Flatten()(pool2)

    # Fully Connected Layer 1
    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer='he_normal')(flatten)

    # Fully Connected Layer 2
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer='he_normal')(fc1)

    # Output Layer
    output = K.layers.Dense(units=10, activation='softmax')(fc2)

    model = K.models.Model(inputs=X, outputs=output)

    # Compile the model
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
