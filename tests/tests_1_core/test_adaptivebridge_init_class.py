#!/bin/env python
# tests/tests_1_core/test_adaptivebridge_init_class.py

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
import copy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import adaptivebridge.adaptivebridge as adaptivebridge

# Test case for initializing the AdaptiveBridge class with default parameters


def test_init():
    # Create an instance of AdaptiveBridge with a LinearRegression model
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())
    # Create a deepcopy of the LinearRegression model for comparison
    model = copy.deepcopy(LinearRegression())

    # Assert that the model type is the same, but not the same instance
    assert type(ab.model) == type(
        model) and ab.model != model and ab.model is not model

    # Assert that default parameters are set correctly
    assert ab.correlation_threshold == adaptivebridge.CORRELATION_THRESHOLD
    assert ab.importance_threshold == adaptivebridge.IMPORTANCE_THRESHOLD
    assert ab.min_accuracy == adaptivebridge.MIN_ACCURACY
    assert ab.default_accuracy_selection == adaptivebridge.DEFAULT_ACCURACY_SELECTION
    assert ab.accuracy_logic == adaptivebridge.ACCURACY_LOGIC
    assert ab.feature_engineering == None
    assert ab.x_df == None
    assert ab.y_df == None
    assert ab.x_df_columns == None
    assert ab.feature_distribution == (None)
    assert ab.feature_importance == None
    assert ab.feature_map == None
    assert ab.corr_matrix == None
    assert ab.max_feature == None
    assert ab.max_index == (None)
    assert ab.model_map == None
    assert ab.training_time == None

# Test case for initializing the AdaptiveBridge class with specified parameters


def test__with_parameters_init():
    # Create an instance of AdaptiveBridge with specified parameters
    ab = adaptivebridge.AdaptiveBridge(LinearRegression(
        fit_intercept=True), 0.5, 0.8, 0.9, 0.2, r2_score)
    # Create a deepcopy of the LinearRegression model for comparison
    model = copy.deepcopy(LinearRegression(fit_intercept=True))

    # Assert that the model type is the same, but not the same instance
    assert type(ab.model) == type(
        model) and ab.model != model and ab.model is not model

    # Assert that specified parameters are set correctly
    assert ab.correlation_threshold == 0.5
    assert ab.min_accuracy == 0.8
    assert ab.default_accuracy_selection == 0.9
    assert ab.importance_threshold == 0.2
    assert ab.accuracy_logic == r2_score
    assert ab.feature_engineering == None
    assert ab.x_df == None
    assert ab.y_df == None
    assert ab.x_df_columns == None
    assert ab.feature_distribution == (None)
    assert ab.feature_importance == None
    assert ab.feature_map == None
    assert ab.corr_matrix == None
    assert ab.max_feature == None
    assert ab.max_index == (None)
    assert ab.model_map == None
    assert ab.training_time == None

# Test case for string representation of the AdaptiveBridge class


def test_str():
    # Create an instance of AdaptiveBridge with default parameters
    ab = adaptivebridge.AdaptiveBridge(LinearRegression())

    # Assert that the string representation matches the expected output
    assert print(ab) == print(f"""
        AdaptiveBridge Class:
         - Parameters:
            - Model (Backbone) = {ab.model.__class__.__name__}
            - Correlation Threshold = {ab.correlation_threshold}
            - Minimum Accuracy = {ab.min_accuracy}
            - Default Accuracy Selection = {ab.default_accuracy_selection}
            - Importance Threshold = {ab.importance_threshold}
            - Accuracy Logic = {ab.accuracy_logic.__name__}
         - Model:
            - Trained = {ab.model_map is not None}
            - Training UTC Time = {ab.training_time}
            """)


# Run pytest when the script is executed
if __name__ == "__main__":
    pytest.main()
