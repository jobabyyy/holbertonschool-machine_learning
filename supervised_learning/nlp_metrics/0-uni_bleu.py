#!/usr/bin/env python3
"""NLP - Evaluation Metrics:
Calculating Unigram BLEU Score."""


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
    # Calculate precision for each reference
    precisions = []
    sentence_words = set(sentence)

    for ref in references:
        common_words = sentence_words.intersection(ref)
        precision = len(common_words) / len(sentence)
        precisions.append(precision)

    # Calculate brevity penalty
    close_l = min(references, key=lambda
                  ref: abs(len(ref) - len(sentence)))
    close_l = len(close_l)
    brev_pen = 1 if len(sentence) >= close_l else len(sentence) / close_l

    # Calculate BLEU score
    product = 1 if not precisions else (sum(precisions) / len(precisions))
    bleu = brev_pen * product

    return bleu
