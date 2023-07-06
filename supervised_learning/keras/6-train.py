#!/usr/bin/env python3
"""trains a model using gradient descent
part3 continued..."""

import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent
    and analyzes vaildation data"""
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop_callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
        callbacks.append(early_stop_callback)

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose, shuffle=shuffle)
    return history
