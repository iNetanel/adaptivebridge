#!/bin/env python
# tests/tests_0_utils/test_utils_continuous_central_tendency.py

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
from adaptivebridge.utils._data_distribution import _continuous_central_tendency
from unittest.mock import patch  # Import patch from unittest.mock

# Test cases for <-0.5 skewness and norm.e


def test_empty_mode():
    # Create a pandas Series
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5])
    # Mock the _calculate_central_tendency function to return 2
    # Mock the _choose_central_tendency function to return "mean"
    with patch('adaptivebridge.utils._data_distribution._calculate_central_tendency', return_value=2), \
            patch('adaptivebridge.utils._data_distribution._choose_central_tendency', return_value="mean"):
        # Call the _continuous_central_tendency function
        result = _continuous_central_tendency(data)
    # Assert the distribution type is one of the expected values
    assert result[0] in ["norm", "expon", "gamma", "dweibull", "lognorm",
                         "pareto", "t", "beta", "uniform", "loggamma", "genextreme"]
    # Assert the scale parameter matches the mocked value
    assert result[1] == 2
    # Assert the central tendency method matches the mocked value
    assert result[2] == "mean"


if __name__ == "__main__":
    # Run the tests using pytest
    pytest.main()
