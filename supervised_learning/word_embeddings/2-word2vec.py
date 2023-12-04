#!/usr/bin/env python3
"""Model that trains gensim word2vec model."""


from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True,
                   iterations=5, seed=0, workers=1):
    """
    Create and train a gensim word2vec model.

    Args:
        sentences: List of sentences to be trained on.
        size: Dimensionality of the embedding layer.
        min_count: Minimum number of occurrences
                   of a word for use in training.
        window: Maximum distance between the current
                and predicted word within a sentence.
        negative: Size of negative sampling.
        cbow: Boolean to determine the training type;
              True is for CBOW; False is for Skip-gram.
        iterations: Number of iterations to train over.
        seed: Seed for the random number generator.
        workers: Number of worker threads to train the model.

    Returns:
        The trained Word2Vec model.
    """
    if cbow:
        sg = 0  # CBOW
    else:
        sg = 1  # Skip-gram

    model = Word2Vec(
        sentences=sentences,
        vector_size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,  # training type
        epochs=iterations,  # Use epochs instead of iter
        seed=seed,
        workers=workers
    )

    return model
