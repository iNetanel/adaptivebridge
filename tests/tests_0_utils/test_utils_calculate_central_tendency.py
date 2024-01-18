#!/bin/env python
# tests/tests_0_utils/test_utils_calculate_central_tendency.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import pandas as pd
import pytest
from adaptivebridge.utils._data_distribution import _calculate_central_tendency

# Test cases for _calculate_central_tendency function


def test_calculate_central_tendency_mean():
    # Create a pandas Series
    data = pd.Series([1, 2, 3, 4, 5])
    # Call the _calculate_central_tendency function with "mean"
    result = _calculate_central_tendency("mean", data)
    # Assert the result matches the expected mean value
    assert result == 3.0


def test_calculate_central_tendency_median():
    # Create a pandas Series
    data = pd.Series([1, 2, 3, 4, 5])
    # Call the _calculate_central_tendency function with "median"
    result = _calculate_central_tendency("median", data)
    # Assert the result matches the expected median value
    assert result == 3.0


def test_calculate_central_tendency_mode():
    # Create a pandas Series
    data = pd.Series([1, 2, 3, 3, 3, 4, 4])
    # Call the _calculate_central_tendency function with "mode"
    result = _calculate_central_tendency("mode", data)
    # Assert the result matches the expected mode value
    assert result == 3


def test_calculate_central_tendency_default():
    # Create a pandas Series
    data = pd.Series([1, 2, 3, 4, 5])
    # Call the _calculate_central_tendency function with an unknown method
    result = _calculate_central_tendency("unknown_method", data)
    # Assert the result matches the default value (assuming it is the first element in the Series)
    assert result == 1


if __name__ == "__main__":
    # Run the tests using pytest
    pytest.main()
