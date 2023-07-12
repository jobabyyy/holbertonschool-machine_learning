#!/usr/bin/env python3
"""function that performs forward propagation over a
convolutional layer of a neural network"""

import numpy as np

import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """"doc"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                        (pad_w, pad_w), (0, 0)), mode="constant")

    h_out = int((h_prev - kh + 2 * pad_h) / sh) + 1
    w_out = int((w_prev - kw + 2 * pad_w) / sw) + 1

    Z = np.zeros((m, h_out, w_out, c_new))
    for c in range(c_new):
        for h in range(h_out):
            for w in range(w_out):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw
                
                a_slice_prev = A_prev_pad[
                    :, vert_start:vert_end, horiz_start:horiz_end, :]
                Z[:, h, w, c] = np.sum(
                    a_slice_prev * W[:, :, :, c], axis=(1, 2, 3)) + b[0, 0, 0, c]

    A = activation(Z)

    return A
