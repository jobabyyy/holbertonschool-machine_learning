#!/usr/bin/env python3
"""
Transformers App - 
Task 3: Pipeline.
"""


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


class Dataset:
    """
    Class that loads and preps a dataset for machine translation.
    """  
    def __init__(self, batch_size, max_len):
        """
        Update class constructor __init__:
        batch_size: the batch size for the training/validation
        max_len: the max num of tokens allowed per sentence
        UPDATE: data_train attribute by performing the following actions:
                1: filter out all examples that have either sentence w/more
                   than max_len tokens
                2: cache the dataset to increase performance
                3: shuffle the entire dataset
                4: split the dataset into padded batches of size batch_size
                5: prefetch the dataset using tf.data.experimental.AUTOTUNE
                   performance       
        """
        # load training and validation
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)

        # build subword tokenizers for Portuguese and English
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.data_train),
            target_vocab_size=2**15
        )

        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.data_train),
            target_vocab_size=2**15
        )

        # filter, cache, shuffle, batch, and prefetch training dataset
        self.data_train = self.data_train.filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len, tf.size(en) <= max_len))
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(tf.data.experimental.cardinality(self.data_train).numpy())
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        # filter and batch the validation dataset
        self.data_valid = self.data_valid.filter(lambda pt, en: tf.logical_and(tf.size(pt) <= max_len, tf.size(en) <= max_len))
        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.padded_batch(batch_size)

    def tf_encode(self, pt, en):
        pt, en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        pt.set_shape([None])
        en.set_shape([None])
        return pt, en

    def encode(self, pt, en):
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())

        # add start and end tokens
        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + [self.tokenizer_en.vocab_size + 1]

        return np.array(pt_tokens), np.array(en_tokens)

if __name__ == '__main__':
  import tensorflow as tf

tf.compat.v1.set_random_seed(0)
data = Dataset(32, 40)
for pt, en in data.data_train.take(1):
    print(pt, en)
for pt, en in data.data_valid.take(1):
    print(pt, en)
