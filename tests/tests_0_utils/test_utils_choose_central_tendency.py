#!/bin/env python
# tests/tests_0_utils/test_utils_choose_central_tendency.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import pandas as pd
from unittest.mock import patch  # Import patch from unittest.mock
import pytest
from adaptivebridge.utils._data_distribution import _choose_central_tendency

# Test cases for >0.5 skewness and norm.e list


def test_big_skewness():
    # Create a pandas Series
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])
    # Mock the skew function to return a positive skewness value
    with patch('adaptivebridge.utils._data_distribution.skew', return_value=1):
        # Call the _choose_central_tendency function with "norm" distribution
        result = _choose_central_tendency("norm", data)
    # Assert the result matches the expected choice
    assert result == "median"

# Test cases for <-0.5 skewness and norm.e


def test_negative_skewness():
    # Create a pandas Series
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])
    # Mock the skew function to return a negative skewness value
    with patch('adaptivebridge.utils._data_distribution.skew', return_value=-1):
        # Call the _choose_central_tendency function with "norm" distribution
        result = _choose_central_tendency("norm", data)
    # Assert the result matches the expected choice
    assert result == "median"

# Test cases for <-0.5 skewness and norm.e


def test_with_mode():
    # Create a pandas Series
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])
    # Mock the mode function to return a Series with values [2, 3]
    # Mock the skew function to return 0 (for simplicity)
    with patch('pandas.Series.mode', return_value=pd.Series([2, 3])), \
            patch('adaptivebridge.utils._data_distribution.skew', return_value=0):
        # Call the _choose_central_tendency function with "norm" distribution
        result = _choose_central_tendency("norm", data)
    # Assert the result matches the expected choice
    assert result == "mode"

# Test cases for <-0.5 skewness and norm.e


def test_empty_mode():
    # Create a pandas Series
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])
    # Mock the mode function to return an empty Series
    # Mock the skew function to return 0 (for simplicity)
    with patch('pandas.Series.mode', return_value=pd.Series([])), \
            patch('adaptivebridge.utils._data_distribution.skew', return_value=0):
        # Call the _choose_central_tendency function with "norm" distribution
        result = _choose_central_tendency("norm", data)
    # Assert the result matches the expected choice
    assert result == "mean"

# Test cases for non norm.e list


def test_non_norm_e_list():
    # Create a pandas Series
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])
    # Mock the skew function to return 0 (for simplicity)
    with patch('adaptivebridge.utils._data_distribution.skew', return_value=0):
        # Call the _choose_central_tendency function with "expon" distribution
        result = _choose_central_tendency("expon", data)
    # Assert the result matches the expected choice
    assert result == "median"

# Test cases for not supported distribution


def test_not_supported_distribution():
    # Create a pandas Series
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])
    # Call the _choose_central_tendency function with an unsupported distribution
    result = _choose_central_tendency("not_in_list", data)
    # Assert the result matches the default choice
    assert result == "mean"


if __name__ == "__main__":
    # Run the tests using pytest
    pytest.main()
