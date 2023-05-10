#!/usr/bin/env python3
"""func to add 2 array element-wise"""


def add_arrays(arr1, arr2):
    """Check that the input arrays have the same length"""

    if len(arr1) != len(arr2):
        return None
    
    # Create a new array to hold the sums
    result = []
    
    # Add the corresponding elements of the input arrays and append the sums to the result array
    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    
    # Return the result array
    return result
