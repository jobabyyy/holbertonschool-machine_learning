#!/usr/bin/env python3
"""performs pooling on images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images
    images.
    Returns: numpy.ndarray containing the pooled images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pooled_h = (h - kh) // sh + 1
    pooled_w = (w - kw) // sw + 1

    pooled_images = np.zeros((m, pooled_h, pooled_w, c))

    for i in range(pooled_h):
        for j in range(pooled_w):
            window = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw]

            if mode == 'max':
                pooled_images[:, i, j] = np.max(window, axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j] = np.mean(window, axis=(1, 2))

    return pooled_images
