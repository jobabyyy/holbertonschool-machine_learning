#!/usr/bin/env python3
"""NLP - Evaluation Metrics:
Calculating Unigram BLEU Score."""


from nltk.translate.bleu_score import sentence_bleu


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
    score = sentence_bleu(references, sentence)

    return (score)
