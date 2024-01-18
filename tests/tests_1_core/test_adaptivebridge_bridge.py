#!/bin/env python
# tests/tests_1_core/test_adaptivebridge_bridge.py

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
import re
import adaptivebridge.adaptivebridge as adaptivebridge
from adaptivebridge.utils import MandatoryFeatureError
from sklearn.linear_model import LinearRegression


@pytest.fixture
def test_data():
    # Load test data from CSV file
    data = pd.read_csv('./tests/test_data.csv', header=0, sep=',')
    # Separate dependent and independent variables
    y_df = data["weight"]
    x_df = data.drop(["weight"], axis=1)

    return {
        "x_df": x_df,
        "y_df": y_df
    }

# Test case for a pandas Series with a single mode


def test_bridge(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    # Fit the model with the test data
    ab.fit(test_data["x_df"], test_data["y_df"])
    # Perform bridging operation on x_df
    new_x_df = ab.bridge(test_data["x_df"])

    # Assertions
    assert not new_x_df.isnull().values.any()
    assert not new_x_df.isnull().any(axis=0).any()
    assert new_x_df.size == test_data["x_df"].size


def test_bridge_missing_full_feature_deviation(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and custom min_accuracy
    ab = adaptivebridge.AdaptiveBridge(LinearRegression(), min_accuracy=0.001)
    # Fit the model with the test data
    ab.fit(test_data["x_df"], test_data["y_df"])
    # Remove 'radius' feature from x_df
    x_df = test_data["x_df"].drop(["radius"], axis=1)
    # Perform bridging operation on modified x_df
    new_x_df = ab.bridge(x_df)

    # Assertions
    assert not new_x_df.isnull().values.any()
    assert not new_x_df.isnull().any(axis=0).any()
    assert new_x_df.size == test_data["x_df"].size


def test_bridge_missing_partial_feature_deviation(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and custom min_accuracy
    ab = adaptivebridge.AdaptiveBridge(LinearRegression(), min_accuracy=0.001)
    # Fit the model with the test data
    ab.fit(test_data["x_df"], test_data["y_df"])
    # Set some values in 'radius' feature to NaN
    x_df = test_data["x_df"].copy()
    x_df.at[1, 'radius'] = np.NaN
    x_df.at[5, 'radius'] = np.NaN
    # Perform bridging operation on modified x_df
    new_x_df = ab.bridge(x_df)

    # Assertions
    assert not new_x_df.isnull().values.any()
    assert not new_x_df.isnull().any(axis=0).any()
    assert new_x_df.size == test_data["x_df"].size


def test_bridge_missing_full_feature_adaptive(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and custom min_accuracy, correlation_threshold
    ab = adaptivebridge.AdaptiveBridge(
        LinearRegression(), min_accuracy=0.001, correlation_threshold=0.01)
    # Fit the model with the test data
    ab.fit(test_data["x_df"], test_data["y_df"])
    # Remove 'circum' feature from x_df
    x_df = test_data["x_df"].drop(["circum"], axis=1)
    # Perform bridging operation on modified x_df
    new_x_df = ab.bridge(x_df)

    # Assertions
    assert not new_x_df.isnull().values.any()
    assert not new_x_df.isnull().any(axis=0).any()
    assert new_x_df.size == test_data["x_df"].size


def test_bridge_missing_partial_feature_adaptive(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and custom min_accuracy, correlation_threshold
    ab = adaptivebridge.AdaptiveBridge(
        LinearRegression(), min_accuracy=0.001, correlation_threshold=0.01)
    # Fit the model with the test data
    ab.fit(test_data["x_df"], test_data["y_df"])
    # Set some values in 'circum' feature to NaN
    x_df = test_data["x_df"].copy()
    x_df.at[1, 'circum'] = np.NaN
    x_df.at[5, 'circum'] = np.NaN
    # Perform bridging operation on modified x_df
    new_x_df = ab.bridge(x_df)

    # Assertions
    assert not new_x_df.isnull().values.any()
    assert not new_x_df.isnull().any(axis=0).any()
    assert new_x_df.size == test_data["x_df"].size


def test_missing_mandatory_feature_bridge(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and custom min_accuracy
    ab = adaptivebridge.AdaptiveBridge(LinearRegression(), min_accuracy=0.001)
    x_df = test_data["x_df"]

    # Assertions using pytest.raises
    with pytest.raises(MandatoryFeatureError, match="A mandatory feature is completely missing: circum"):
        ab.fit(x_df, test_data["y_df"])
        x_df = x_df.drop(["circum"], axis=1)
        ab.bridge(x_df)

    with pytest.raises(MandatoryFeatureError, match=re.escape("A mandatory feature is partially missing: circum > (please check for NaN values in your dataset)")):
        x_df = test_data["x_df"].copy()
        x_df.at[1, 'circum'] = np.NaN
        x_df.at[5, 'circum'] = np.NaN
        ab.bridge(x_df)


if __name__ == "__main__":
    pytest.main()
