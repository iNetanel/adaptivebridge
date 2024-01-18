#!/bin/env python
# tests/tests_0_utils/test_utils_high_level_distribution.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import pytest
from adaptivebridge.utils._data_distribution import _high_level_distribution

# Test case for continuous distribution


def test_continuous_distribution():
    """
    Test case for identifying a continuous distribution.
    """
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    result = _high_level_distribution(data)
    assert result == "continuous"

# Test case for discrete distribution due to threshold


def test_continuous_distribution_not_meet_threshold():
    """
    Test case for identifying a discrete distribution when not meeting the threshold.
    """
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = _high_level_distribution(data)
    assert result == "discrete"

# Test case for discrete distribution


def test_discrete_distribution():
    """
    Test case for identifying a discrete distribution.
    """
    data = [1, 1, 2, 2, 3, 3, 1, 2, 3, 1, 2, 3, 4, 1, 2]
    result = _high_level_distribution(data)
    assert result == "discrete"

# Test case for edge case with only one unique value


def test_one_unique_value():
    """
    Test case for handling an edge case with only one unique value.
    """
    data = [1, 1, 1, 1, 1]
    result = _high_level_distribution(data)
    assert result == "discrete"

# Test case for edge case with a single element


def test_single_element():
    """
    Test case for handling an edge case with a single element.
    """
    data = [42]
    result = _high_level_distribution(data)
    assert result == "discrete"

# Test case for edge case with zero average distance


def test_zero_avg_distance():
    """
    Test case for handling an edge case with zero average distance.
    """
    data = [1, 1, 1, 1, 1]
    result = _high_level_distribution(data)
    assert result == "discrete"

# Test case for a case where discrete distribution should be identified due to low number of unique values for float


def test_flexible_discrete_not_meet_avg_distance():
    """
    Test case for identifying a discrete distribution based on average distance.
    """
    data = [1, 1.2, 1.4, 1.6, 1.8, 2]
    result = _high_level_distribution(data)
    assert result == "discrete"

# Test case for a case where continuous distribution should be identified due to avg distance


def test_flexible_continuous_meet_avg_distance():
    """
    Test case for identifying a continuous distribution based on average distance.
    """
    data = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
            2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
    result = _high_level_distribution(data)
    assert result == "continuous"

# Test case for a case where discrete distribution should be identified due to avg distance


def test_flexible_continuous_below_avg_distance():
    """
    Test case for identifying a discrete distribution based on average distance.
    """
    data = [2, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4.75, 5, 5]
    result = _high_level_distribution(data)
    assert result == "discrete"


if __name__ == "__main__":
    pytest.main()
