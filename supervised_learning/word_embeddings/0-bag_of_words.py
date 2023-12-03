#!/usr/bin/env python3
"""NLP - Word Embedding:
Bag of Words."""


import numpy as np


def bag_of_words(sentences, vocab=None):
    """Function: bag of words
    function is to create bag of words
    embedding matrix.
    
    Args:
         Sentences: list of sentences
                    to analyze
         Vocab: list of the vocab words
                within sentences to use
                for the analysis.
    Returns:
            Embeddings: numpy.ndarray
                        of shape (s, f)
                        containing
                        embeddings
                        - S: # of sentences
                        - F: # of features
            Feautures: list of the features
                       used for embeddings."""

    # convert sentences to lowercase
    sentences = [sentence.lower() for sentence in sentences]

    # Create vocabulary if not provided
    if vocab is None:
        vocab = set()
        for sentence in sentences:
            words = sentence.split()
            vocab.update(words)

    # create embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)))
    # create features list
    features = list(vocab)

    # embeddings matrix
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for j, feature in enumerate(features):
            embeddings[i, j] = words.count(feature)

    return embeddings, features
