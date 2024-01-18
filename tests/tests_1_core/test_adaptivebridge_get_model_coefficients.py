#!/bin/env python
# tests/tests_1_core/test_adaptivebridge_get_model_coefficients.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

import pandas as pd
import pytest
import adaptivebridge.adaptivebridge as adaptivebridge
from adaptivebridge.utils import MandatoryFeatureError
import re

# Import necessary libraries
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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


def test_get_model_coefficients_coefficients(test_data):
    # Create an instance of AdaptiveBridge with LinearRegression model
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    # Fit the model with test data
    ab.fit(test_data["x_df"], test_data["y_df"])
    # Get model coefficients
    result = ab._get_model_coefficients()
    assert result is not None


def test_get_model_coefficients_with_classifier(test_data):
    # Create an instance of AdaptiveBridge with LogisticRegression model
    ab = adaptivebridge.AdaptiveBridge(LogisticRegression())
    # Modify y_df to have binary values for classification
    y_df = test_data["y_df"]
    y_df.iloc[:100] = 1
    y_df.iloc[100:] = 0
    # Fit the model with modified test data
    ab.model.fit(test_data["x_df"], y_df)
    # Get model coefficients
    result = ab._get_model_coefficients()
    assert result is not None


def test_get_model_coefficients_feature_importances(test_data):
    # Create an instance of AdaptiveBridge with RandomForestClassifier model
    ab = adaptivebridge.AdaptiveBridge(RandomForestClassifier())
    # Modify y_df to have binary values for classification
    y_df = test_data["y_df"]
    y_df.iloc[:100] = 1
    y_df.iloc[100:] = 0
    # Fit the model with modified test data
    ab.model.fit(test_data["x_df"], y_df)
    # Get model coefficients
    result = ab._get_model_coefficients()
    assert result is not None


def test_get_model_coefficients_feature_importances_error_raised(test_data):
    # Create an instance of AdaptiveBridge with KNeighborsClassifier model
    ab = adaptivebridge.AdaptiveBridge(KNeighborsClassifier())
    # Modify y_df to have binary values for classification
    y_df = test_data["y_df"]
    y_df.iloc[:100] = 1
    y_df.iloc[100:] = 0
    # Fit the model with modified test data
    ab.model.fit(test_data["x_df"], test_data["y_df"])
    # Check if ValueError is raised for an unsupported model type
    with pytest.raises(ValueError, match=re.escape(f"Model type {type(ab.model)} not recognized or supported")):
        _ = ab._get_model_coefficients()


if __name__ == "__main__":
    pytest.main()
