#!/usr/bin/env python3
"""using custom padding"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding
    images.
    Returns: numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (
                                    ph, ph), (pw, pw)), mode='constant')

    # Perform the convolution
    convolved_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            window = padded_images[:, i:i+kh, j:j+kw]
            convolved_images[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return convolved_images
