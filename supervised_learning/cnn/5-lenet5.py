#!/usr/bin/env python3
"""func that builds a modified version
of the LeNet-5 architecture
using Keras"""


import tensorflow.keras as K


def lenet5(X):
    # Convolutional layer 1
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            activation='relu',
                            kernel_initializer='he_normal')(X)
    # Max pooling layer 1
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    # Convolutional layer 2
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                            activation='relu',
                            kernel_initializer='he_normal')(pool1)
    # Max pooling layer 2
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    # Flatten the input for fully connected layers
    flatten = K.layers.Flatten()(pool2)
    # Fully connected layer 1
    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer='he_normal')(flatten)
    # Fully connected layer 2
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer='he_normal')(fc1)
    # Output softmax layer
    output = K.layers.Dense(units=10, activation='softmax')(fc2)

    model = K.Model(inputs=X, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
