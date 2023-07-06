#!/usr/bin/env python3
"""func to determine if u
should stop gradient descent early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """should we stop?..."""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count >= patience:
        return True, count

    return False, count
