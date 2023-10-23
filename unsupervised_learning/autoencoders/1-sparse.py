#!/usr/bin/env python3
"""Autoencoder: Sparse Autoencoder"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    input_dims: is an integer containing the dimensions of
                the model input
    hidden_layers: list containing the number of nodes for each
                    hidden layer in the encoder, respectively
                   - the hidden layers should be reversed for the decoder
    latent_dims: is an integer containing the dimensions of the
                 latent space representation
    lambtha: is the regularization parameter used for L1 regularization
             on the encoded output
    Returns: encoder, decoder, auto
             - encoder: is the encoder model
             - decoder: is the decoder model
             - auto: is the sparse autoencoder model
    """
    encoder_input = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0],
                                 activation='relu')(encoder_input)

    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(encoded)
    encoded = keras.layers.Dense(latent_dims,
                                 activation='relu',
                                 activity_regularizer=keras.regularizers.l1(
                                    lambtha)
                                 )(encoded)

    # Decoder
    input_decoder = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(input_decoder)

    for i in range(len(hidden_layers)-2, -1, -1):
        decoded = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(decoded)

    decoded = keras.layers.Dense(input_dims,
                                 activation='sigmoid')(decoded)

    # Models
    encoder = keras.models.Model(inputs=encoder_input, outputs=encoded)
    decoder = keras.models.Model(inputs=input_decoder, outputs=decoded)
    auto = keras.models.Model(inputs=encoder_input,
                              outputs=decoder(encoder(encoder_input)))

    # Compile Model
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
