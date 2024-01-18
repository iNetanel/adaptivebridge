#!/bin/env python
# tests/tests_1_core/test_adaptivebridge_fit.py

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
import re
import adaptivebridge.adaptivebridge as adaptivebridge
from adaptivebridge.utils import EngineeringFeatureError, _convert_to_dataframe
from sklearn.linear_model import LinearRegression
import re


@pytest.fixture
def test_data():
    # Load test data from CSV file
    data = pd.read_csv('./tests/test_data.csv', header=0, sep=',')
    # Separate dependent and independent variables
    y_df = data["weight"]
    x_df = data.drop(["weight"], axis=1)
    # Initialize AdaptiveBridge with LinearRegression
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())

    return {
        "x_df": x_df,
        "y_df": y_df,
        "ab": ab
    }

# Test case for fitting the model with default parameters


def test_fit(test_data):
    # Fit the model with the test data
    test_data["ab"].fit(test_data["x_df"], test_data["y_df"])
    # Convert input data to DataFrame and Series for comparison
    x_df = _convert_to_dataframe(test_data["x_df"], "dataframe")
    y_df = _convert_to_dataframe(test_data["y_df"], "series")

    # Assert that internal attributes are updated as expected
    assert test_data["ab"].x_df is None
    assert test_data["ab"].y_df is None
    assert test_data["ab"].x_df_columns.equals(x_df.columns)
    assert test_data["ab"].feature_engineering is not None
    assert test_data["ab"].feature_distribution is not (None)
    assert test_data["ab"].feature_importance is not None
    assert test_data["ab"].feature_map is not None
    assert test_data["ab"].corr_matrix is not None
    assert test_data["ab"].max_feature is not None
    assert test_data["ab"].max_index is not (None)
    assert test_data["ab"].model_map is not None
    assert test_data["ab"].training_time is not None

# Test case for fitting the model with additional feature engineering


def test_feature_engineering_fit(test_data):
    # Modify x_df to include an additional feature "radius2"
    x_df = test_data["x_df"]
    y_df = test_data["y_df"]
    x_df["radius2"] = x_df["radius"]**2
    # Fit the model with the modified test data and specified feature engineering
    test_data["ab"].fit(x_df, y_df, feature_engineering=["radius2"])
    # Convert input data to DataFrame and Series for comparison
    x_df = _convert_to_dataframe(x_df, "dataframe")
    y_df = _convert_to_dataframe(y_df, "series")

    # Assert that internal attributes are updated as expected
    assert test_data["ab"].x_df is None
    assert test_data["ab"].y_df is None
    assert test_data["ab"].x_df_columns.equals(x_df.columns)
    assert test_data["ab"].feature_engineering is not None
    assert test_data["ab"].feature_distribution is not (None)
    assert test_data["ab"].feature_importance is not None
    assert test_data["ab"].feature_map is not None
    assert test_data["ab"].corr_matrix is not None
    assert test_data["ab"].max_feature is not None
    assert test_data["ab"].max_index is not (None)
    assert test_data["ab"].model_map is not None
    assert test_data["ab"].training_time is not None

# Test case for fitting the model and handling missing feature in feature engineering


def test_feature_engineering_fit_missing(test_data):
    # Fit the model with the original test data
    test_data["ab"].fit(test_data["x_df"], test_data["y_df"])

    # Attempt fitting with missing user-defined feature, expecting an error
    with pytest.raises(EngineeringFeatureError, match="User-defined feature-engineering feature required and is completely missing: radius2"):
        test_data["ab"].fit(test_data["x_df"], test_data["y_df"],
                            feature_engineering=["radius2"])

    # Modify x_df to include an additional feature "radius2" with a missing value
    x_df = test_data["x_df"]
    x_df["radius2"] = x_df["radius"]**2
    x_df.at[1, 'radius2'] = pd.NA

    # Attempt fitting with a missing value in the user-defined feature, expecting an error
    with pytest.raises(EngineeringFeatureError, match=re.escape("User-defined feature-engineering feature is partially missing: radius2 > (please check for NaN values in your dataset)")):
        test_data["ab"].fit(x_df, test_data["y_df"],
                            feature_engineering=["radius2"])


if __name__ == "__main__":
    pytest.main()
