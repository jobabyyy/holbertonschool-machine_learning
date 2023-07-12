#!/usr/bin/env python3
"""func that performs backward prop
over a convolutional layer"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Retrieving dimensions from input"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    h_out = dZ.shape[1]
    w_out = dZ.shape[2]

    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Padding
    if padding == "same":
        pad_h = int(np.ceil((h_prev * sh - sh + kh - h_prev) / 2))
        pad_w = int(np.ceil((w_prev * sw - sw + kw - w_prev) / 2))
        A_prev_pad = np.pad(
            A_prev,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant')
        dA_prev_pad = np.pad(
            dA_prev,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
            mode='constant')
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end,
                                         horiz_start:horiz_end, :]
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end,
                                :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
        # Gradients
        if padding == "same":
            dA_prev[i, :, :, :] = da_prev_pad[pad_h:-pad_h, pad_w:-pad_w, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    return dA_prev, dW, db
