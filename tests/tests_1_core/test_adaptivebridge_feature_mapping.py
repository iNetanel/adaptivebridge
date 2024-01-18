#!/bin/env python
# tests/tests_1_core/test_adaptivebridge_feature_mapping.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import pytest
import adaptivebridge.adaptivebridge as adaptivebridge
from sklearn.linear_model import LinearRegression
from unittest.mock import patch  # Import patch from unittest.mock


def test_feature_mapping():
    # Create an instance of AdaptiveBridge with LinearRegression model
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())

    # Mock necessary methods to return "OK" for testing purposes
    with patch('adaptivebridge.AdaptiveBridge._mandatory_and_distribution', return_value="OK"), \
            patch('adaptivebridge.AdaptiveBridge._adaptive_model', return_value="OK"):

        # Run the feature_mapping method
        ab._feature_mapping()

        # Define the expected feature map
        feature_map = {
            "engineering": {},
            "mandatory": {},
            "deviation": {},
            "adaptive": {}
        }

        # Assert that the feature map matches the expected result
        assert feature_map == ab.feature_map


if __name__ == "__main__":
    pytest.main()
