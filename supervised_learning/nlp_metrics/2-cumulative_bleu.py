#!/usr/bin/env python3
"""Bleu Score: Cumulative N-Gram"""


import numpy as np
from collections import Counter


def calculate_precision(candidate, references, n):
    """
    Precision calculation function
    """
    cand_ngrams = Counter(zip(*[candidate[i:] for i in range(n)]))
    ref_ngrams = Counter()

    for ref in references:
        ref_ngrams += Counter(zip(*[ref[i:] for i in range(n)]))

    clipped_counts = {ngram: min(cand_ngrams[ngram],
                      ref_ngrams[ngram]) for ngram in cand_ngrams}
    precision = sum(clipped_counts.values()
                    ) / max(1, sum(cand_ngrams.values()))

    return precision


def cumulative_bleu(references, sentence, n):
    """Function cumulative bleu"""
    total_precision = 1.0

    for i in range(1, n + 1):
        precision_i = calculate_precision(sentence, references, i)
        total_precision *= precision_i

    closest_ref = min(references, key=lambda
                      ref: abs(len(ref) - len(sentence)))
    brev_penalty = min(1, len(sentence) / len(closest_ref))

    cumu_score = brev_penalty * total_precision ** (1/n)

    return cumu_score
