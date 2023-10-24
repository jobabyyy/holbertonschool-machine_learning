#!/usr/bin/env python3
"""Autoencoder: Variational"""


import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
        - input_dims: integer, dimensions of the model input.
        - hidden_layers: list, number of nodes for each hidden
          layer in the encoder.
        - latent_dims: integer, dimensions of the latent
          space representation.

    Returns:
        - encoder: the encoder model.
        - decoder: the decoder model.
        - auto: the full autoencoder model.
    """
    # Encoder
    inputs = tf.keras.layers.Input(shape=(input_dims,))
    x = inputs

    for units in hidden_layers:
        x = tf.keras.layers.Dense(units, activation='relu')(x)

    mean_layer = tf.keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = tf.keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        mean_layer, log_var = args
        batch = tf.shape(mean_layer)[0]
        dim = tf.shape(mean_layer)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean_layer + tf.exp(0.5 * log_var) * epsilon

    z = tf.keras.layers.Lambda(sampling,
                               output_shape=(latent_dims,
                                             ))([mean_layer, log_var])

    encoder = tf.keras.models.Model(inputs, [z, mean_layer, log_var])

    # Decoder
    decoder_inputs = tf.keras.layers.Input(shape=(latent_dims,))
    x = decoder_inputs

    for units in reversed(hidden_layers):
        x = tf.keras.layers.Dense(units, activation='relu')(x)

    decoder_outputs = tf.keras.layers.Dense(input_dims,
                                            activation='sigmoid')(x)

    decoder = tf.keras.models.Model(decoder_inputs, decoder_outputs)

    # Autoencoder
    encoder_outputs = encoder(inputs)[0]
    decoder_outputs = decoder(encoder_outputs)
    auto = tf.keras.models.Model(inputs, decoder_outputs)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
