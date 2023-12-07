#!/usr/bin/env python3
"""NLP - Evaluation Metrics:
Calculating Unigram BLEU Score."""


import numpy as np


def uni_bleu(references, sentence):
    """
    Function that calculates the unigram
    BLEU score for a sentence:

    Args:
         references: list of reference
                     translations...
                     - each translation of
                     translation is a list of
                     the words in the translatiom.
         sentence: list containing the model
                   proposed sentence.
    Returns:
            the unigram BLEU score.
    """
    # initialize directories
    word_counts = {}
    max_ref_counts = {}

    # max count of each word in references
    for reference in references:
        for word in reference:
            current_count = reference.count(word)
            max_ref_counts[word] = max(max_ref_counts.get(word, 0),
                                       current_count)

    # count each word in the sentence
    for word in sentence:
        if word in max_ref_counts:
            word_counts[word] = min(word_counts.get(word, 0) + 1,
                                    max_ref_counts[word])
    # calculate precision
    precision = sum(word_counts.values()) / len(sentence)

    # calculate length of closest reference to sentence
    closest_ref = min(references, key=lambda
                      ref: abs(len(ref) - len(sentence)))
    closest_ref = len(closest_ref)

    # Calculate brevity penalty to penalize overly short translations
    if len(sentence) > closest_ref:
        brev_penality = 1
    else:
        brev_penality = np.exp(1 - float(closest_ref) / len(sentence))

    # Calculate BLEU score
    bleu_score = brev_penality * precision

    return bleu_score
