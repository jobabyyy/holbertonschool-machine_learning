#!/usr/bin/env python3
"""Autoencoder: Vanilla
function that creates an
autoencoder."""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    input_dims: is an integer containing the dimensions
                of the model input
    hidden_layers: list containing the number of nodes
                   for each hidden layer in the encoder,
                   respectively
                   - the hidden layers should be reversed
                     for the decoder
    latent_dims: is an integer containing the dimensions
                 of the latent space representation
    Returns: encoder, decoder, auto
             - encoder: is the encoder model
             - decoder: is the decoder model
             - auto: is the full autoencoder model
    """
    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))

    x = encoder_input
    for layer_dims in hidden_layers:
        x = keras.layers.Dense(layer_dims,
                               activation='relu')(x)
    encoder_output = keras.layers.Dense(latent_dims,
                                        activation='relu')(x)

    encoder = keras.Model(encoder_input, encoder_output)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for layer_dims in reversed(hidden_layers):
        x = keras.layers.Dense(layer_dims, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    # Autoencoder
    autoencoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(autoencoder_input, decoded)

    # Compile Autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
