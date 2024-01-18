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

# Test case for a pandas Series with a single mode


def test__adaptive_predict(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and specified parameters
    ab = adaptivebridge.AdaptiveBridge(
        LinearRegression(), min_accuracy=0.001, correlation_threshold=0.01)
    # Fit the model with the test data
    ab.fit(test_data["x_df"], test_data["y_df"])
    # Prepare adaptive prediction input data
    adap_x_df = test_data["x_df"].drop(["iron_type", "circum", "area"], axis=1)
    adap_y_df = test_data["x_df"]["circum"]
    # Perform adaptive prediction
    adaptive_predict = ab._adaptive_predict(adap_x_df, "circum")
    # Assert the size of the result matches the size of the ground truth
    assert adaptive_predict.size == adap_y_df.size

# Test case for a pandas Series with a single mode


def test__adaptive_predict_second_case(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model and specified parameters
    ab = adaptivebridge.AdaptiveBridge(
        LinearRegression(), min_accuracy=0.001, correlation_threshold=0.01)
    # Prepare input data by dropping certain columns
    x_df = test_data["x_df"].drop(["radius", "diameter"], axis=1)
    # Fit the model with the modified data
    ab.fit(x_df, test_data["y_df"])
    # Prepare adaptive prediction input data with a single feature
    adap_x_df = pd.DataFrame({"parimeter": test_data["x_df"]["parimeter"]})
    adap_y_df = test_data["x_df"]["circum"]
    # Perform adaptive prediction
    adaptive_predict = ab._adaptive_predict(adap_x_df, "circum")
    # Assert the size of the result matches the size of the ground truth
    assert adaptive_predict.size == adap_y_df.size

# Test case for a pandas Series with a single mode ##### disable due to warnings of pandas.
# def test__adaptive_predict_1d_case(test_data):
#     ab = adaptivebridge.AdaptiveBridge(LinearRegression(), min_accuracy=0.001, correlation_threshold=0.01)
#     x_df = test_data["x_df"].drop(["radius", "diameter"], axis=1)
#     ab.fit(x_df, test_data["y_df"])
#     adap_x_df = pd.DataFrame({"parimeter": test_data["x_df"]["parimeter"]})
#     adap_x_df_series = adap_x_df["parimeter"]
#     adap_y_df = test_data["x_df"]["circum"]
#     adaptive_predict = ab._adaptive_predict(adap_x_df_series, "circum")
#     assert adaptive_predict.size == adap_y_df.size


if __name__ == "__main__":
    pytest.main()
