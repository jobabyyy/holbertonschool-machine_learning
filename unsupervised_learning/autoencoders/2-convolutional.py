#!/usr/bin/env python3
"""Autoencoder: Convolutional
                Autoencoder """


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    input_dims: is a tuple of integers containing
                the dimensions of the model input
    filters: is a list containing the number of filters
             for each convolutional layer in the encoder,
             respectively
                - the filters should be reversed for the decoder
    latent_dims: is a tuple of integers containing the
                 dimensions of the latent space representation
    Each convolution in the encoder should use a kernel size of
    (3, 3) with same padding and relu activation, followed by
    max pooling of size (2, 2)
    Each convolution in the decoder, except for the last two,
    should use a filter size of (3, 3) with same padding and
    relu activation, followed by upsampling of size (2, 2)
            - The second to last convolution should instead
              use valid padding
            - The last convolution should have the same
              number of filters as
              the number of channels in input_dims with
              sigmoid activation
              and no upsampling
    Returns: encoder, decoder, auto
                - encoder: is the encoder model
                - decoder: is the decoder model
                - auto: is the full autoencoder model
    """
    # Input layer
    encoder_input = keras.layers.Input(shape=input_dims)
    x = encoder_input

    # Convolutional layers
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Shape
    non_flat = keras.backend.int_shape(x)[1:]

    # latent vector
    x = keras.layers.Flatten()(x)
    encoder_output = keras.layers.Dense(latent_dims[0] *
                                        latent_dims[1] * latent_dims[2],
                                        activation='relu')(x)

    # Encoder model
    encoder = keras.models.Model(encoder_input, encoder_output)

    # Input layer
    input_shape = (latent_dims[0] * latent_dims[1] * latent_dims[2],)
    decoder_input = keras.layers.Input(shape=input_shape)

    x = keras.layers.Dense(input_shape)(decoder_input)
    x = keras.layers.Reshape(non_flat)(x)

    # Convolutional layers
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu',
                            padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Output layer w/ sigmoid
    decoder_output = keras.layers.Conv2D(input_dims[2], (3, 3),
                                         activation='sigmoid',
                                         padding='same')(x)

    # Decoder model
    decoder = keras.models.Model(decoder_input, decoder_output)

    autoencoder_output = decoder(encoder(encoder_input))
    auto = keras.models.Model(encoder_input, autoencoder_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
