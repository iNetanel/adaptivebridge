#!/bin/env python
# tests/tests_0_utils/test_utils_mode_selector.py

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
from adaptivebridge.utils._data_distribution import _mode_selector

# Test case for a pandas Series with a single mode


def test_single_mode():
    """
    Test case for a pandas Series with a single mode.
    """
    data = [1, 2, 2, 3, 3, 3, 4]
    series = pd.Series(data)
    assert _mode_selector(series) == 3

# Test case for a pandas Series with multi modes


def test_multi_mode():
    """
    Test case for a pandas Series with multiple modes.
    """
    data = [1, 2, 3, 3, 3, 4, 4, 4]
    series = pd.Series(data)
    assert _mode_selector(series) == 3

# Test case for a pandas Series with no mode


def test_no_mode():
    """
    Test case for a pandas Series with no mode.
    """
    data = [1, 2, 3, 4, 5]
    series = pd.Series(data)
    assert _mode_selector(series) is None

# Test case for a boolean pandas Series with a tie


def test_boolean_tie():
    """
    Test case for a boolean pandas Series with a tie.
    """
    data = [True, True, False, False]
    series = pd.Series(data)
    assert _mode_selector(series) == True

# Test case for a boolean pandas Series with a single mode


def test_boolean_single_mode():
    """
    Test case for a boolean pandas Series with a single mode.
    """
    data = [True, True, False, False, False]
    series = pd.Series(data)
    assert _mode_selector(series) == False


if __name__ == "__main__":
    pytest.main()
