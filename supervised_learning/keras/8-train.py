#!/usr/bin/env python3
"""trains a model to save best interation
part4 continued..."""

import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1,
                verbose=True, shuffle=False):
    """updates function to also
    save the best iteration of the model"""
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop_callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
        callbacks.append(early_stop_callback)

    if learning_rate_decay and validation_data is not None:
        def lr_decay(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay_callback = K.callbacks.LearningRateScheduler(
            lr_decay, verbose=1)
        callbacks.append(lr_decay_callback)

    if save_best and filepath is not None:
        model_checkpoint = K.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', save_best_only=True)
        callbacks.append(model_checkpoint)

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose, shuffle=shuffle)

    return history
