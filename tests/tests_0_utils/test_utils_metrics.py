#!/bin/env python
# tests/tests_0_utils/test_utils_metrics.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import numpy as np
import pytest

from adaptivebridge.utils._metrics import _percentage_error, _mean_absolute_percentage_error


@pytest.fixture
def pred_data_and_actual():
    """
    Fixture providing actual and predicted data for testing metrics.
    """
    return {
        "actual": np.array([10, 20, 0, 30, 40]),
        "predicted": np.array([12, 18, 5, 28, 38])}


def test_percentage_error(pred_data_and_actual):
    """
    Test case for the _percentage_error function.
    """
    result = _percentage_error(
        pred_data_and_actual["actual"], pred_data_and_actual["predicted"])
    expected_result = np.array([-0.2, 0.1, 0.25, 0.06666667, 0.05])
    np.testing.assert_allclose(result, expected_result, rtol=1e-2, atol=1e-2)


def test_mean_absolute_percentage_error(pred_data_and_actual):
    """
    Test case for the _mean_absolute_percentage_error function.
    """
    result = _mean_absolute_percentage_error(
        pred_data_and_actual["actual"], pred_data_and_actual["predicted"])
    expected_result = 0.13333333333333336
    np.testing.assert_allclose(result, expected_result, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main()
