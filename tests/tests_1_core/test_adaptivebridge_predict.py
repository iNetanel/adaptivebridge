#!/bin/env python
# tests/tests_2_e2e/test_adaptivebridge.py

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
from sklearn.linear_model import LinearRegression
import adaptivebridge.adaptivebridge as adaptivebridge
from adaptivebridge.utils import EngineeringFeatureError

# Test data fixture


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

# Test case for predicting with AdaptiveBridge


def test_predict(test_data):
    # Initialize AdaptiveBridge with LinearRegression
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())

    # Fit the model with the training data
    ab.fit(test_data["x_df"], test_data["y_df"])

    # Predict using the trained model
    ypred = ab.predict(test_data["x_df"])

    # Assert that there are no NaN values in the predictions
    assert not np.isnan(ypred).any()

    # Assert that the size of predictions matches the size of the target variable
    assert ypred.size == test_data["y_df"].size

# Test case for feature engineering with missing values


def test_feature_engineering_predict_missing(test_data):
    # Initialize AdaptiveBridge with LinearRegression
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())

    # Create a new feature "radius2" by squaring the "radius" feature
    x_df = test_data["x_df"]
    x_df["radius2"] = x_df["radius"] ** 2

    # Fit the model with the new feature and target variable
    ab.fit(x_df, test_data["y_df"], feature_engineering=["radius2"])

    # Drop the newly created feature "radius2" and attempt prediction, expecting an error
    with pytest.raises(EngineeringFeatureError, match="User-defined feature-engineering feature required and is completely missing: radius2"):
        test_data["x_df"] = test_data["x_df"].drop(["radius2"], axis=1)
        ab.predict(test_data["x_df"])

    # Re-add the "radius2" feature and introduce a missing value, expecting an error
    x_df = test_data["x_df"]
    x_df["radius2"] = x_df["radius"] ** 2
    with pytest.raises(EngineeringFeatureError, match=re.escape("User-defined feature-engineering feature is partially missing: radius2 > (please check for NaN values in your dataset)")):
        x_df.at[1, 'radius2'] = pd.NA
        ab.predict(x_df)


# Run pytest when the script is executed
if __name__ == "__main__":
    pytest.main()
