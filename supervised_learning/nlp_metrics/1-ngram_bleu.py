#!/usr/bin/env python3
"""N-Gram BLEU Score"""


import numpy as np
from collections import Counter


def ngram_bleu(references, sentence, n):
    """
    references: is a list of reference translations
                - each reference translation is a list of
                  the words in the translation
    sentence: is a list containing the model proposed sentence
    n: is the size of the n-gram to use for evaluation

    Returns: the n-gram BLEU score
    """
    def get_ngrams(tokens, n):
        """
        function get_ngrams
        """
        return [tuple(tokens[i:i + n])
                for i in range(len(tokens) - n + 1)]

    can_ngrams = Counter(get_ngrams(sentence, n))
    ref_ngrams = Counter()

    for ref in references:
        ref_ngrams += Counter(get_ngrams(ref, n))

    clipped_counts = {ngram: min(can_ngrams[ngram],
                      ref_ngrams[ngram]) for ngram in can_ngrams}

    precision = sum(clipped_counts.values()
                    ) / max(1, sum(can_ngrams.values()))

    closest_ref = min(references, key=lambda
                      ref: abs(len(ref) - len(sentence)))
    brev_penalty = np.exp(1 - (len(closest_ref) / len(sentence))
                          ) if len(sentence) < len(closest_ref) else 1.0

    bleu = brev_penalty * precision

    return bleu

