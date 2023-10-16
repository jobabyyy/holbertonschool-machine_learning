#!/usr/bin/env python3
"""func that builds a modified version
of the LeNet-5 architecture
using Keras"""

import tensorflow as tf

# Set the random seed for TensorFlow
SEED = 0
tf.random.set_seed(SEED)

def lenet5(X):
    """building a modified version of LeNet-5"""
    init = K.initializers.he_normal(seed=None)

    # convolutional layer 1
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            activation='relu', kernel_initializer=init)(X)

    # Pooling
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                            activation='relu',
                            kernel_initializer=init)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flatten = K.layers.Flatten()(pool2)

    # fully connected layer1
    fc1 = K.layers.Dense(units=120, activation='relu',
                         kernel_initializer=init)(flatten)
    # fully connected layer2
    fc2 = K.layers.Dense(units=84, activation='relu',
                         kernel_initializer=init)(fc1)
    # output layer
    output = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=init)(fc2)

    model = K.Model(inputs=X, outputs=output)
    # model complied
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
