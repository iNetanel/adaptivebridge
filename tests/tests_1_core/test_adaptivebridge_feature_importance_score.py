#!/bin/env python
# tests/tests_1_core/test_adaptivebridge_feature_importance_score.py

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


def test_get_model_feature_importance_score(test_data, capsys):
    # Create an instance of AdaptiveBridge with LinearRegression model
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    # Fit the model with the test data
    ab.fit(test_data["x_df"], test_data["y_df"])
    # Calculate and print feature importance scores
    ab.feature_importance_score(test_data["x_df"])
    # Capture the printed output to check or assert later
    captured = capsys.readouterr()

    # Add assertions based on the expected output
    # Replace these with the actual values or patterns you expect
    assert "Feature:" in captured.out


if __name__ == "__main__":
    pytest.main()
