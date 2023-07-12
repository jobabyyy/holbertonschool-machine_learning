#!/usr/bin/env python3
"""func that perfroms backprop
over a pooling layer"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Retrieving dimensions from input"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    dA_prev = np.zeros_like(A_prev)

    """performing backward propagation"""
    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                for c in range(c_prev):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_prev_slice = A_prev[i, vert_start:vert_end,
                                          horiz_start:horiz_end, c]

                    if mode == 'max':
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, vert_start:vert_end,
                                horiz_start:horiz_end, c] += mask * dA[
                                    i, h, w, c]
                    elif mode == 'avg':
                        average = dA[i, h, w, c] / (kh * kw)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end,
                                c] += np.ones((kh, kw)) * average

    return dA_prev
