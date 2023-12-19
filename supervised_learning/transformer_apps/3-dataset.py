#!/usr/bin/env python3
"""
Imports
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """
    Dataset Class
    """
    def __init__(self, batch_size, max_len):
        """
        Init Function
        """
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en', split='validation', as_supervised=True)

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        self.data_train = self.data_train.filter(lambda x, y: tf.logical_and(tf.size(x) <= max_len, tf.size(y) <= max_len))
        self.data_valid = self.data_valid.filter(lambda x, y: tf.logical_and(tf.size(x) <= max_len, tf.size(y) <= max_len))

        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(buffer_size=10000)
        self.data_train = self.data_train.padded_batch(batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.padded_batch(batch_size, padded_shapes=([None], [None]))


    def tokenize_dataset(self, data):
        """
        Tokenize Dataset Function
        """
        all_text_pt = []
        all_text_en = []
        for pt, en in data:
            all_text_pt.append(pt.numpy())
            all_text_en.append(en.numpy())

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            all_text_pt, target_vocab_size=2**15
        )

        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            all_text_en, target_vocab_size=2**15
        )

        return tokenizer_pt, tokenizer_en


    def encode(self, pt, en):
        """
        Encode Function
        """

        ptA = self.tokenizer_pt.encode(pt.numpy())
        ptB = [self.tokenizer_pt.vocab_size + 1]

        enA = self.tokenizer_en.encode(en.numpy())
        enB = [self.tokenizer_en.vocab_size + 1]

        pt = [self.tokenizer_pt.vocab_size] + ptA + ptB
        en = [self.tokenizer_en.vocab_size] + enA + enB

        return pt, en


    def tf_encode(self, pt, en):
        """
        TensorFlow Wrapper For Encode Function
        """
        pt_tensor, en_tensor = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])

        pt_tensor.set_shape([None])
        en_tensor.set_shape([None])

        return pt_tensor, en_tensor