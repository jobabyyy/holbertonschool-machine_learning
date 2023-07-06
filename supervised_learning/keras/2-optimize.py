#!/usr/bin/env python3
"""function that sets up Adam opt
for a keras model w/categorical
crossentropy loss and accuracy metrics"""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """adam opt set up for keras model"""
    metrics = ['accuracy']
    loss = 'categorical_crossentropy'

    opt = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=opt, loss=loss, metrics=metrics)
