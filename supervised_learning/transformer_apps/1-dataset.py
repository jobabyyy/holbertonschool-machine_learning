#!/usr/bin/env python3
"""
Transformers App -
Taks 1: Encode Tokens
"""


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np



class Dataset:
    """
    Class that loads and preps a dataset for machine translation: 
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

        # load training and validation
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)

        # build subword tokenizers for Portuguese and English
        self.tokenizer_pt = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.data_train),
            target_vocab_size=2**15
        ))
        self.tokenizer_en = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.data_train),
            target_vocab_size=2**15
        ))

    def tokenize_dataset(self, data):
        # retrieve the tokenizers created during init
        tokenizer_pt = self.tokenizer_pt
        tokenizer_en = self.tokenizer_en

        def encode(pt, en):
            # tokenize sentences
            pt_tokens = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(pt.numpy()) + [tokenizer_pt.vocab_size]
            en_tokens = [tokenizer_en.vocab_size] + tokenizer_en.encode(en.numpy()) + [tokenizer_en.vocab_size]

            return np.array(pt_tokens), np.array(en_tokens)

        # apply encode function to each element of dataset
        data = data.map(encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return data

    def tokenize_dataset(self, data):
        # retrieve the tokenizers created during init
        tokenizer_pt = self.tokenizer_pt
        tokenizer_en = self.tokenizer_en

    def encode(self, pt, en):
        # use the tokenizer instances to encode the sentences
        pt_tokens = self.tokenizer_pt.encode(pt.numpy())
        en_tokens = self.tokenizer_en.encode(en.numpy())

        # add start and end tokens
        pt_tokens = [self.tokenizer_pt.vocab_size] + pt_tokens + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + en_tokens + [self.tokenizer_en.vocab_size + 1]

        return np.array(pt_tokens), np.array(en_tokens)

# testing our output
if __name__ == "__main__":
  import tensorflow as tf

  data = Dataset()
  for pt, en in data.data_train.take(1):
      print(data.encode(pt, en))
  for pt, en in data.data_valid.take(1):
      print(data.encode(pt, en))
