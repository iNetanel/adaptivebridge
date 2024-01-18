#!/bin/env python
# adaptivebridge/utils/_metrics.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import numpy as np

# Helper for the Mean absolute percentage error (MAPE) regression loss


def _percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

# Mean absolute percentage error (MAPE) regression loss


def _mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(_percentage_error(np.asarray(y_true), np.asarray(y_pred))))
