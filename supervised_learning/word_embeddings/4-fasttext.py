#!/usr/bin/env python3
"""Function that creates and
trains a genism fasttext Model:"""


from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Create and train a Gensim FastText model.

    Args:
        sentences: List of sentences to be trained on.
        size: Dimensionality of the embedding layer.
        min_count: Minimum number of occurrences of a
                   word for use in training.
        negative: Size of negative sampling.
        window: Maximum distance between the current and
                predicted word within a sentence.
        cbow: Boolean to determine the training type
              True for CBOW, False for Skip-gram).
        epochs: Number of training epochs.
        seed: Seed for the random number generator.
        workers: Number of worker threads to train the model.

    Returns:
        The trained FastText model.
    """
    model = FastText(
        sentences=sentences,
        vector_size=size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=0 if cbow else 1,  # 0 for CBOW, 1 for Skip-gram
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
