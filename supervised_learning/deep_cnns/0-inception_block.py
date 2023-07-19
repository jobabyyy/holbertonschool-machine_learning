#!/usr/bin/env python3
"""Function to create the inception
block and constructs a Keras
Model with specific input
and output tensors"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """filters added to layers"""

    F1, F3R, F3, F5R, F5, FPP = filters

    # convolution branch
    branch_1x1 = K.layers.Conv2D(F1, (1, 1),
                                 activation='relu')(A_prev)

    # convolution before 3x3 convolution branch
    branch_3x3 = K.layers.Conv2D(F3R, (1, 1),
                                 activation='relu')(A_prev)
    branch_3x3 = K.layers.Conv2D(F3, (3, 3), padding='same',
                                 activation='relu')(branch_3x3)

    # convolution before 5x5 convolution branch
    branch_5x5 = K.layers.Conv2D(F5R, (1, 1),
                                 activation='relu')(A_prev)
    branch_5x5 = K.layers.Conv2D(F5, (5, 5), padding='same',
                                 activation='relu')(branch_5x5)

    # Max pooling branch
    branch_pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                        padding='same')(A_prev)
    branch_pool = K.layers.Conv2D(FPP, (1, 1),
                                  activation='relu')(branch_pool)

    # concat branches
    output = K.layers.concatenate([branch_1x1, branch_3x3,
                                  branch_5x5, branch_pool], axis=-1)

    return output
