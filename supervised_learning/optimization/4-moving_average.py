#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set.
"""


def moving_average(data, beta):
    """calc weight and returns moving averages"""
    moving_averages = []
    avg = 0
    bias_correction = 1 - beta

    for i, value in enumerate(data):
        avg = (beta * avg + (1 - beta) * value) / bias_correction
        moving_averages.append(avg)

    return moving_averages
