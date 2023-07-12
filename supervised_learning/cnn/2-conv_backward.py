#!/usr/bin/env python3
"""func that performs backward prop
over a convolutional layer"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Retrieving dimensions from input"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # computing dimensions of output volume after padding
    if padding == "same":
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                                 (pad_w, pad_w), (0, 0)), mode="constant")

    h_out = int((h_prev - kh + 2 * pad_h) / sh) + 1
    w_out = int((w_prev - kw + 2 * pad_w) / sw) + 1

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice_prev = A_prev_pad[
                        i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # computing gradients using the chain rule
                    dA_prev[i, vert_start:vert_end,
                            horiz_start:horiz_end, :] += W[
                                           :, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice_prev * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

    # adjusting the gradients
    if padding == "same":
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]

    return dA_prev, dW, db
