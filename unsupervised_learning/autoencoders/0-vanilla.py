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
    input_layer = keras.layers.Input(shape=(input_dims,))
    encoder_layers = []

    for num_nodes in hidden_layers:
        encoder_layers.append(keras.layers.Dense(num_nodes,
                              activation='relu')(input_layer))
        input_layer = encoder_layers[-1]

    latent_layer = keras.layers.Dense(latent_dims,
                                      activation='relu')(input_layer)
    
    # Dense layer to transform the encoder
    transformed = keras.layers.Dense(hidden_layers[-1],
                                     activation='relu')(latent_layer)
    
    # Encoder model with transformed output
    encoder = keras.models.Model(inputs=input_layer,
                                 outputs=transformed)

    # Decoder
    input_layer = keras.layers.Input(shape=(hidden_layers[-1],))
    decoder_layers = []

    for num_nodes in reversed(hidden_layers):
        decoder_layers.append(keras.layers.Dense(num_nodes,
                              activation='relu')(input_layer))
        input_layer = decoder_layers[-1]

    output_layer = keras.layers.Dense(input_dims,
                                      activation='sigmoid')(input_layer)

    decoder = keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Autoencoder
    autoencoder = keras.models.Model(inputs=encoder.input,
                                     outputs=decoder(encoder.output))

    # Compile
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
