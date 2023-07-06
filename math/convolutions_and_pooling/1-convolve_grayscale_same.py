#!/usr/bin/env python3
"""performs a same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images
    images.
    Returns: numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the padding size
    pad_h = kh // 2
    pad_w = kw // 2

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (
                                    pad_w, pad_w)), mode='constant')

    # Perform the convolution
    convolved_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            window = padded_images[:, i:i+kh, j:j+kw]
            convolved_images[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return convolved_images
