#!/bin/env python
# tests/tests_0_utils/test_utils_discrete_central_tendency.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import pytest
from adaptivebridge.utils._data_distribution import _discrete_central_tendency

# Parametrized test cases covering different scenarios


@pytest.mark.parametrize("data, expected_result", [
    # Test case: All values are the same
    (pd.Series([1, 1, 1, 1]), ("discrete", 1, "constant")),

    # Test case: All unique values
    (pd.Series([1, 2, 3, 4, 5]), ("discrete", 3, "median")),

    # Test case: Mode is most frequent
    (pd.Series([1, 2, 2, 3, 3, 3]), ("discrete", 3, "mode")),

    # Test case: Default to median
    (pd.Series([1, 2, 3, 4, 4]), ("discrete", 3, "median")),
])
def test_discrete_central_tendency(data, expected_result):
    result = _discrete_central_tendency(data)
    assert result == expected_result

# Additional test cases for edge scenarios, NaN handling, etc.


def test_discrete_central_tendency_with_nan():
    # Test case: NaN handling
    data = pd.Series([1, 2, np.nan, 4, 5])
    result = _discrete_central_tendency(data)
    assert result == ("discrete", 3, "median")

# Additional test cases for edge scenarios, NaN handling, etc.


def test_discrete_central_tendency_with_nan_and_bool():
    # Test case: NaN handling
    data = pd.Series([0, 0, 1, 1])
    data = data.astype(bool)
    result = _discrete_central_tendency(data)
    assert result == ('discrete', False, 'mode')


def test_discrete_central_tendency_bool_FT():
    # Test case: Boolean data
    data = pd.Series([True, True, False, True])
    result = _discrete_central_tendency(data)
    assert result == ("discrete", True, "mode")


def test_discrete_central_tendency_with_bool():
    # Test case: Boolean data
    data = pd.Series([True, True, False, False])
    data = data.astype(bool)
    result = _discrete_central_tendency(data)
    assert result == ("discrete", True, "mode")


def test_discrete_central_tendency_rounding():
    # Test case: Rounding for integer-like data
    data = pd.Series([1, 2, 3, 4])
    result = _discrete_central_tendency(data)
    assert result == ("discrete", 2, "median")


if __name__ == "__main__":
    pytest.main()
