#!/bin/env python
# tests/tests_1_core/test_adaptivebridge_feature_sequence.py

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
import matplotlib.pyplot as plt
from unittest.mock import patch  # Import patch from unittest.mock


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


def test_feature_sequence_without_none(test_data, capsys, monkeypatch):

    # Mock plt.show() to prevent plots from being displayed
    def mock_show(*args, **kwargs):
        pass

    monkeypatch.setattr(plt, 'show', mock_show)

    # Create an instance of AdaptiveBridge with LinearRegression model
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    x_df = test_data["x_df"]
    x_df["radius2"] = x_df["radius"]**2
    # Fit the model with the test data and specified feature engineering
    ab.fit(test_data["x_df"], test_data["y_df"],
           feature_engineering=["radius2"])

    # Run the feature_sequence function
    ab.feature_sequence()

    # Capture the printed output to check or assert later
    captured = capsys.readouterr()

    # Add assertions based on the expected output
    # Replace these with the actual values or patterns you expect
    assert "Feature Sequence Dependencies" in captured.out
    assert "User-defined feature-engineering features: (Must be provided by the user)" in captured.out
    assert "Mandatory: (Must be provided by the user)" in captured.out
    assert "Data Distribution Method: (data distribution method will be used and not prediction)" in captured.out
    assert "Prediction by Adaptive Model: (will be predict by adaptiv model)" in captured.out


def test_feature_sequence_with_none(test_data, capsys, monkeypatch):

    # Mock plt.show() to prevent plots from being displayed
    def mock_show(*args, **kwargs):
        pass

    monkeypatch.setattr(plt, 'show', mock_show)

    # Create an instance of AdaptiveBridge with LinearRegression model and customized parameters
    ab = adaptivebridge.AdaptiveBridge(LinearRegression(
    ), correlation_threshold=1, min_accuracy=1, importance_threshold=1)
    # Fit the model with the test data
    ab.fit(test_data["x_df"], test_data["y_df"])

    # Run the feature_sequence function
    ab.feature_sequence()

    # Capture the printed output to check or assert later
    captured = capsys.readouterr()

    # Add assertions based on the expected output
    # Replace these with the actual values or patterns you expect
    assert "Feature Sequence Dependencies" in captured.out
    assert "User-defined feature-engineering features: (Must be provided by the user)\n - None" in captured.out
    assert "Mandatory: (Must be provided by the user)\n - None" in captured.out
    assert "Data Distribution Method: (data distribution method will be used and not prediction)" in captured.out
    assert "Prediction by Adaptive Model: (will be predict by adaptiv model)\n - None" in captured.out


def test_feature_sequence_without_mandatory(test_data, capsys, monkeypatch):

    # Mock plt.show() to prevent plots from being displayed
    def mock_show(*args, **kwargs):
        pass

    monkeypatch.setattr(plt, 'show', mock_show)

    # Create an instance of AdaptiveBridge with LinearRegression model and customized parameters
    ab = adaptivebridge.AdaptiveBridge(LinearRegression(
    ), correlation_threshold=0, min_accuracy=0, importance_threshold=0)
    # Fit the model with the test data
    ab.fit(test_data["x_df"], test_data["y_df"])

    # Run the feature_sequence function
    ab.feature_sequence()

    # Capture the printed output to check or assert later
    captured = capsys.readouterr()

    # Add assertions based on the expected output
    # Replace these with the actual values or patterns you expect
    assert "Feature Sequence Dependencies" in captured.out
    assert "User-defined feature-engineering features: (Must be provided by the user)" in captured.out
    assert "Mandatory: (Must be provided by the user)" in captured.out
    assert "Data Distribution Method: (data distribution method will be used and not prediction)\n - None" in captured.out
    assert "Prediction by Adaptive Model: (will be predict by adaptiv model)" in captured.out


if __name__ == "__main__":
    pytest.main()
