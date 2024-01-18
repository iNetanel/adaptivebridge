#!/bin/env python
# tests/tests_1_core/test_adaptivebridge_calculate_importance_and_distribution.py

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


def test_get_model_calculate_importance_and_distribution(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    # Fit the model with the test data
    ab.model.fit(test_data["x_df"], test_data["y_df"])
    # Set x_df and feature_distribution attributes
    ab.x_df = test_data["x_df"]
    ab.feature_distribution = ab._distribution()
    # Calculate and return feature importance
    result = ab._calculate_importance()
    # Assert that the result is not None
    assert result is not None


if __name__ == "__main__":
    pytest.main()
