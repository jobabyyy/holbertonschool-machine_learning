#!/usr/bin/env python3
"""
Transformer App - 
Task 4: Create Masks.
"""


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np


def create_masks(inputs, target):
    """
    inputs: tf.Tensor of shape (batch_size, seq_len_in) that contains
            the input sentence
    target: tf.Tensor of shape (batch_size, seq_len_out) that contains
            the target sentence
    This function should only use tensorflow operations in order to properly
    function in the training step

    Returns: encoder_mask, combined_mask, decoder_mask
    encoder_mask: is the tf.Tensor padding mask of shape
                  (batch_size, 1, 1, seq_len_in) to be applied in the encoder
    combined_mask: is the tf.Tensor of shape (batch_size, 1, seq_len_out, seq_len_out)
                   used in the 1st attention block in the decoder to
                   pad and mask future tokens in the input received by the decoder.
                   It takes the maximum between a lookaheadmask and the
                   decoder target padding mask.
    decoder_mask: is the tf.Tensor padding mask of shape
                  (batch_size, 1, 1, seq_len_in) used in the 2nd attention
                  block in the decoder.
    """
    # encoder padding mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # used in the 2nd attention block in the decoder
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    look_ahead_mask = tf.linalg.band_part(tf.ones((target.shape[1], target.shape[1])), -1, 0)
    decoder_target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_target_padding_mask = decoder_target_padding_mask[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask


if __name__ == '__main__':

  import tensorflow as tf

  tf.compat.v1.set_random_seed(0)
  data = Dataset(32, 40)
  for inputs, target in data.data_train.take(1):
      print(create_masks(inputs, target))
      