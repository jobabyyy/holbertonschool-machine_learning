#!/usr/bin/env python3
"""NLP - Word Embedding:
Bag of Words."""


import numpy as np
import re


def tokenize(sentence):
    """Using this function to normalize
    by converting to lowercase and tokenize
    using regex."""

    sentence = sentence.lower()
    words = re.findall(r"\b\w+\b", sentence)
    return words


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
    tokenized_sentences = [tokenize(sentence) for
                           sentence in sentences]

    # print(sentences)

    # Create vocabulary if not provided
    if vocab is None:
        excluded_words = {"s"}
        all_words = [word for sentence in tokenized_sentences
                     for word in sentence]  # flatten sentences

        vocab = sorted(set(word for word in
                       all_words if word not in excluded_words))

    # initialize embeddings matrix
    num_sentences = len(tokenized_sentences)
    num_features = len(vocab)
    embeddings = np.zeros((num_sentences,
                           num_features), dtype=int)

    # Create features list from vocab
    features = list(vocab)

    # fill in embeddings matrix
    for i, sentence in enumerate(tokenized_sentences):
        for j, word in enumerate(features):
            embeddings[i, j] = sentence.count(word)

    return embeddings, features
