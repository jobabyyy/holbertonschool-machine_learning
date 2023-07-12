#!/usr/bin/env python3
"""func that builds a modified version
of the LeNet-5 architecture
using Tensorflow"""


import tensorflow as tf


def lenet5(x, y):
    """building modified version of LeNet-5"""
    initializer = tf.contrib.layers.variance_scaling_initializer()

    # convolutional Layer 1
    conv1 = tf.layers.conv2d(x, filters=6, kernel_size=(5, 5), padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=initializer)

    #  pooling Layer 1
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2))

    # convolutional Layer 2
    conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=(5, 5),
                             padding='valid', activation=tf.nn.relu,
                             kernel_initializer=initializer)

    # pooling Layer 2
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2))

    # Flatten the previous output for the fully connected layers
    flat = tf.layers.flatten(pool2)

    # connected Layer 1
    fc1 = tf.layers.dense(flat, units=120, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # connected Layer 2
    fc2 = tf.layers.dense(fc1, units=84, activation=tf.nn.relu,
                          kernel_initializer=initializer)

    # getting output layer
    output = tf.layers.dense(fc2, units=10, activation=tf.nn.softmax,
                             kernel_initializer=initializer)

    # setting the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                          labels=y, logits=output))

    # setting accuracy
    correct_predictions = tf.equal(tf.argmax(output, axis=1),
                                   tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # training the operation using Adam optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return output, train_op, loss, accuracy
