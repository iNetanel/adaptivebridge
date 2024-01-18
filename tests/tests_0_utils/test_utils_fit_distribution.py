#!/bin/env python
# tests/tests_0_utils/test_utils_fit_distribution.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import pytest
from unittest.mock import patch  # Import patch from unittest.mock
import pandas as pd
from adaptivebridge.utils._data_distribution import _fit_distribution

'''Test cases for _fit_distribution function'''
# Mock functions for discrete and continuous central tendency sets


def mock_discrete_central_tendency(x_df):
    """
    Mock function for discrete central tendency.
    Returns: Tuple containing the best distribution, central value, and measure.
    """
    return "best_dist_discrete", 3, "mode"


def mock_continuous_central_tendency(x_df):
    """
    Mock function for continuous central tendency.
    Returns: Tuple containing the best distribution, central value, and measure.
    """
    return "best_dist_continuous", 5, "mean"

# Test cases for _fit_distribution function


def test_fit_discrete_distribution():
    """
    Test case for fitting a discrete distribution.
    """
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])
    with patch('adaptivebridge.utils._data_distribution._high_level_distribution', return_value="discrete"), \
            patch('adaptivebridge.utils._data_distribution._discrete_central_tendency', side_effect=mock_discrete_central_tendency):
        result = _fit_distribution(data)
    assert result == ["best_dist_discrete", "mode", 3]


def test_fit_continuous_distribution():
    """
    Test case for fitting a continuous distribution.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    with patch('adaptivebridge.utils._data_distribution._high_level_distribution', return_value="continuous"), \
            patch('adaptivebridge.utils._data_distribution._continuous_central_tendency', side_effect=mock_continuous_central_tendency):
        result = _fit_distribution(data)
    assert result == ["best_dist_continuous", "mean", 5]


if __name__ == "__main__":
    pytest.main()
