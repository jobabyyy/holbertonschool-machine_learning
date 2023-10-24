#!/usr/bin/env python3
"""Autoencoder: Convolutional Autoencoder """


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    input_dims: is a tuple of integers containing the
                dimensions of the model input
    filters: is a list containing the number of filters
             for each convolutional layer in the encoder,
             respectively
            - the filters should be reversed for the decoder
    latent_dims: is a tuple of integers containing the
                 dimensions of the latent space representation
    Returns: encoder, decoder, auto
             - encoder: is the encoder model
             - decoder: is the decoder model
             - auto: is the full autoencoder model
    """
    # Encoder
    encoder_input = keras.layers.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2),
                                      padding='same')(x)

    # Encoder model
    encoder = keras.models.Model(encoder_input, x)

    # Decoder
    decoder_input = keras.layers.Input(shape=latent_dims)
    x = decoder_input

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

    # Autoencoder
    autoencoder_output = decoder(encoder(encoder_input))
    auto = keras.models.Model(encoder_input, autoencoder_output)

    # Compile
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto


"""def print_model_summary(encoder, decoder, auto):
    print("Encoder Model Summary:")
    encoder.summary()
    print("\nDecoder Model Summary:")
    decoder.summary()
    print("\nAutoencoder Model Summary:")
    auto.summary()


if __name__ == "__main__":
    # Assuming the dimensions you previously mentioned
    input_dims = (28, 28, 1)
    filters = [16, 8, 8]
    latent_dims = (4, 4, 8)

    encoder, decoder, auto = autoencoder(input_dims, filters, latent_dims)
    print_model_summary(encoder, decoder, auto)"""
