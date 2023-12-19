#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class Dataset:
    def __init__(self):
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='train', as_supervised=True
        ).cache()
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True
        ).cache()

        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.data_train),
            target_vocab_size=2**15
        )

        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.data_train),
            target_vocab_size=2**15
        )

    def tokenize_dataset(self, data):
        tokenizer_pt = self.tokenizer_pt
        tokenizer_en = self.tokenizer_en

        def encode(pt, en):
            pt_tokens = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(pt.numpy()) + [tokenizer_pt.vocab_size + 1]
            en_tokens = [tokenizer_en.vocab_size] + tokenizer_en.encode(en.numpy()) + [tokenizer_en.vocab_size + 1]
            return np.array(pt_tokens), np.array(en_tokens)

        data = data.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return data

    def encode(self, pt, en):
        pt_tokens, en_tokens = (
            self.tokenize_dataset(tf.data.Dataset.from_tensor_slices((pt, en))))
        return pt_tokens, en_tokens
