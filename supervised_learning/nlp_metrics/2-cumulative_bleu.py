import numpy as np
from collections import Counter

def calculate_precision(candidate, references, n):
    candidate_ngrams = Counter(zip(*[candidate[i:] for i in range(n)]))
    reference_ngrams = Counter()

    for ref in references:
        reference_ngrams += Counter(zip(*[ref[i:] for i in range(n)]))

    clipped_counts = {ngram: min(candidate_ngrams[ngram], reference_ngrams[ngram]) for ngram in candidate_ngrams}
    precision = sum(clipped_counts.values()) / max(1, sum(candidate_ngrams.values()))

    return precision

def cumulative_bleu(references, sentence, n):
    total_precision = 1.0

    for i in range(1, n + 1):
        precision_i = calculate_precision(sentence, references, i)
        total_precision *= precision_i

    closest_ref_len = min(len(ref) for ref in references)
    brevity_penalty = min(1, len(sentence) / closest_ref_len)

    cumulative_bleu_score = brevity_penalty * total_precision ** (1/n)

    return cumulative_bleu_score

# Example usage:
references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]
n = 2
print(cumulative_bleu(references, sentence, n))
