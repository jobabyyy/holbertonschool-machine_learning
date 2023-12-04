#!/usr/bin/env python3
"""Model to Keras Embedding layer"""


import numpy as np
from keras.layers import Embedding


def gensim_to_keras(model):
    """
    Convert a gensim word2vec model to a Keras Embedding layer.

    Args:
        model: Trained gensim word2vec model.

    Returns:
        The trainable Keras Embedding layer.
    """
    word_vectors = model.wv
    vocab_size, embedding_size = word_vectors.vectors.shape

    # Create an Embedding layer in Keras
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        weights=[word_vectors.vectors],  # Init weights w/ Gensim model
        input_length=1,  # Each word repped by a single integer index
        trainable=False,  # Make the weights non-trainable
    )

    return embedding_layer
