#usr/bin/env python3
"""
Transformer App -
Task 2: TF Encode.
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

        # load training and validation sets
        data_train_raw = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        data_valid_raw = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)

        # build subword tokenizers for Portuguese and English
        self.tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data_train_raw),
            target_vocab_size=2**15
        )
        self.tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data_train_raw),
            target_vocab_size=2**15
        )

        # tokenize datasets
        self.data_train = data_train_raw.map(self.tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.data_valid = data_valid_raw.map(self.tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def tf_encode(self, pt, en):
        """
        `t_encode` function is responsible for tokenziation and encoding of the text.
        The function's purpose is to serve as a bridge between tensorflows computational
        graph and the custom encode method.
        """
        result_pt, result_en = tf.py_function(func=self.encode, inp=[pt, en], Tout=[tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])
        return result_pt, result_en

    def encode(self, pt, en):
        """
        `encode` function is responsible for converting text data into tokenized and numerical
        format.
        Method takes 2 parameters, `pt`, `en` which rep portuguese and english, respectively.
        """
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(en.numpy()) + [self.tokenizer_en.vocab_size + 1]
        return np.array(pt_tokens), np.array(en_tokens)

if __name__ == "__main__":

  import tensorflow as tf

  data = Dataset()
  # print('got here')
  for pt, en in data.data_train.take(1):
      print(pt, en)
  for pt, en in data.data_valid.take(1):
      print(pt, en)