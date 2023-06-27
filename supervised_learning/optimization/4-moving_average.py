#!/usr/bin/env python3
"""Calculates the weighted moving average of a data set.
"""


def moving_average(data, beta):
    """calc weight and returns moving averages"""
    moving_averages = []
    avg = 0

    for i in range(len(data)):
        avg = beta * avg + (1 - beta) * data[i]
        bias_correction = avg / (1 - beta ** (i + 1))
        moving_averages.append(bias_correction)

    return moving_averages
