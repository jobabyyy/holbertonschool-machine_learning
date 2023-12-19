#!/usr/bin/env python3
"""Transformer App: Dataset"""


import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Class Dataset that loads and preps
    a dataset for machine translation.
    """
    def __init__(self):
        """
        data_train: contains the ted_hrlr_translate/
                    pt_to_en tf.data.Dataset train split,
                    loaded as_supervided
        data_valid: contains the ted_hrlr_translate/
                    pt_to_en tf.data.Dataset validate split,
                    loaded as_supervided
        tokenizer_pt: Portuguese tokenizer created from
                      the training set
        tokenizer_en: English tokenizer created
                      from the training set
        """
        # initializing and loading dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True).cache()
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation',
                                    as_supervised=True).cache()

        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.data_train), target_vocab_size=2**15)

        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.data_train), target_vocab_size=2**15)
         # Fully consume the datasets to avoid caching issues
        self.data_train = self.data_train.reduce(tf.constant(0), lambda x, _: x + 1)
        self.data_valid = self.data_valid.reduce(tf.constant(0), lambda x, _: x + 1)

    def tokenize_dataset(self, data):
        """
        data: is a tf.data.Dataset whose examples are
              formatted as a tuple (pt, en)
                - pt: tf.Tensor containing the Portuguese
                      sentence
                - en: tf.Tensor containing the corresponding
                      English sentence
        The maximum vocab size should be set to 2**15
        Returns:
                tokenizer_pt, tokenizer_en
        tokenizer_pt: Portuguese tokenizer
        tokenizer_en: English tokenizer
        """
        tokenizer_pt = self.tokenizer_pt
        tokenizer_en = self.tokenizer_en

        return tokenizer_pt, tokenizer_en
