#!/usr/bin/env python3
"""perfroms a valid connvolution
on each image using the given kernel"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images
    images.
    Returns: numpy.ndarray containing the convolved images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv_h = h - kh + 1
    conv_w = w - kw + 1

    # Create an empty array to store the convolved images
    convolved_images = np.zeros((m, conv_h, conv_w))

    # Perform the convolution
    for i in range(conv_h):
        for j in range(conv_w):
            window = images[:, i:i+kh, j:j+kw]
            convolved_images[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return convolved_images
