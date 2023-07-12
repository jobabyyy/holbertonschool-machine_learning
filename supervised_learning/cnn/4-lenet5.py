#!/usr/bin/env python3
"""func that builds a modified version
of the LeNet-5 architecture
using Tensorflow"""


import tensorflow as tf


def lenet5(x, y):
    """building modified version of LeNet-5"""
    initializer = tf.contrib.layers.variance_scaling_initializer()

    # convolutional Layer 1
    conv1 = tf.layers.conv2d(
        x, filters=6, kernel_size=5, strides=1,
        padding='same', activation=tf.nn.relu,
        kernel_initializer=initializer)

    # pooling Layer 1
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    # convolutional Layer 2
    conv2 = tf.layers.conv2d(
        pool1, filters=16, kernel_size=5, strides=1,
        padding='valid', activation=tf.nn.relu,
        kernel_initializer=initializer)

    # pooling Layer 2
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # Flatten the pool2 output
    flatten = tf.layers.flatten(pool2)

    # connected Layer 1
    fc1 = tf.layers.dense(
        flatten, units=120,
        activation=tf.nn.relu,
        kernel_initializer=initializer)

    # connected Layer 2
    fc2 = tf.layers.dense(
        fc1, units=84,
        activation=tf.nn.relu,
        kernel_initializer=initializer)

    # getting output layer
    logits = tf.layers.dense(
        fc2, units=10,
        kernel_initializer=initializer)

    # Softmax activation for output
    y_pred = tf.nn.softmax(logits)

    # getting loss function
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y))

    # Accuracy metric
    correct_predictions = tf.equal(
        tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # training operation with Adam optimizer
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    return y_pred, train_op, loss, accuracy
