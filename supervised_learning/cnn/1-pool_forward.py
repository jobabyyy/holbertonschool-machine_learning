#!/usr/bin/env python3
"""Pool forward prop func that
performs fwd prop over a pooling layer"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """retrieve dimensions from input"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    # initialize the output volume with zeros
    A = np.zeros((m, h_out, w_out, c_prev))

    # performing pooling operation
    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_prev):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice_prev = A_prev[
                        i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_slice_prev)
                    elif mode == 'avg':
                        A[i, h, w, c] = np.mean(a_slice_prev)

    return A
