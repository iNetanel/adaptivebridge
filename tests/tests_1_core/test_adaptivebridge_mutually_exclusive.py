#!/bin/env python
# tests/tests_1_core/test_adaptivebridge_adaptive_predict.py

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
import adaptivebridge.adaptivebridge as adaptivebridge
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

# Test case for a mutually exclusive None


def test_mutually_exclusive_diff(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and specified parameters
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    x_df = test_data["x_df"].drop(["DocT", "DocF"], axis=1)
    # Execute mutually exclusive estimator
    ab.x_df = x_df
    ab._mutually_exclusive_detection()
    assert not ab.mutually_exclusive_features_map["diff"]
    assert not ab.mutually_exclusive_features_map["same"]
    assert not ab.mutually_exclusive_features_map["full"]

# Test case for a mutually exclusive diff


def test_mutually_exclusive_diff(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and specified parameters
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    x_df = test_data["x_df"]
    # Execute mutually exclusive estimator
    ab.x_df = x_df
    ab._mutually_exclusive_detection()
    assert ab.mutually_exclusive_features_map["diff"]
    assert ab.mutually_exclusive_features_map["diff"]["DocT"]
    assert ab.mutually_exclusive_features_map["diff"]["DocF"]
    assert not ab.mutually_exclusive_features_map["same"]
    assert ab.mutually_exclusive_features_map["full"]

# Test case for a mutually exclusive same


def test_mutually_exclusive_diff(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and specified parameters
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    x_df = test_data["x_df"]
    x_df["DocT"] = x_df["DocF"]
    # Execute mutually exclusive estimator
    ab.x_df = x_df
    ab._mutually_exclusive_detection()
    assert not ab.mutually_exclusive_features_map["diff"]
    assert ab.mutually_exclusive_features_map["same"]
    assert ab.mutually_exclusive_features_map["same"]["DocT"]
    assert ab.mutually_exclusive_features_map["same"]["DocF"]
    assert ab.mutually_exclusive_features_map["same"]
    assert ab.mutually_exclusive_features_map["full"]

# Test case for a mutually exclusive diff for multi feature


def test_mutually_exclusive_multi_diff(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and specified parameters
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    x_df = test_data["x_df"]
    x_df["DocT2"] = x_df["DocT"]
    # Execute mutually exclusive estimator
    ab.x_df = x_df
    ab._mutually_exclusive_detection()
    assert ab.mutually_exclusive_features_map["diff"]
    assert ab.mutually_exclusive_features_map["diff"]["DocT"]
    assert ab.mutually_exclusive_features_map["diff"]["DocF"]
    assert ab.mutually_exclusive_features_map["diff"]["DocT2"]
    assert ab.mutually_exclusive_features_map["same"]
    assert ab.mutually_exclusive_features_map["full"]

# Test case for a mutually exclusive same for multi feature


def test_mutually_exclusive_same_diff(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and specified parameters
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    x_df = test_data["x_df"]
    x_df["DocT"] = x_df["DocF"]
    x_df["DocT2"] = x_df["DocT"]
    # Execute mutually exclusive estimator
    ab.x_df = x_df
    ab._mutually_exclusive_detection()
    assert not ab.mutually_exclusive_features_map["diff"]
    assert ab.mutually_exclusive_features_map["same"]["DocT2"]
    assert ab.mutually_exclusive_features_map["same"]["DocF"]
    assert ab.mutually_exclusive_features_map["same"]["DocT"]
    assert ab.mutually_exclusive_features_map["full"]


if __name__ == "__main__":
    pytest.main()
