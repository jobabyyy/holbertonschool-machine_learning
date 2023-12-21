#!/usr/bin/env python3
"""Transformer App: Dataset"""


import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    """
    class Dataset that loads and preps a dataset for machine translation:
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

        # Load the dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)

        # Create tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """
        this function's purpose is to build a tokenizer that can convert portuguese
        sentences into sequences of tokens, these tokens are typically numerical.
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=2**15
        )

        # build English tokenizer from the training set
        tokenizer_en = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=2**15
        ))

        return tokenizer_pt, tokenizer_en


if __name__ == "__main__":

  import tensorflow as tf

  data = Dataset()
  for pt, en in data.data_train.take(1):
      print(pt.numpy().decode('utf-8'))
      print(en.numpy().decode('utf-8'))
  for pt, en in data.data_valid.take(1):
      print(pt.numpy().decode('utf-8'))
      print(en.numpy().decode('utf-8'))
  print(type(data.tokenizer_pt))
  print(type(data.tokenizer_en))
