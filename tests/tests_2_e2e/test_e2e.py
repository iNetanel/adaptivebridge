#!/bin/env python
# tests/tests_2_e2e/test_adaptivebridge.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from adaptivebridge.adaptivebridge import AdaptiveBridge

# Fixture to create a sample dataset for testing


@pytest.fixture
def sample_data():
    # Set seed for reproducibility
    np.random.seed(42)
    # Create a DataFrame with random values for features
    X = pd.DataFrame(np.random.rand(100, 5), columns=[
                     'feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
    # Create a Series with random values for the target
    y = pd.Series(np.random.rand(100), name='target')
    return X, y

# Test the basic functionality of AdaptiveBridge


def test_adaptivebridge_basic(sample_data):
    X, y = sample_data
    model = LinearRegression()
    # Create an AdaptiveBridge instance with the LinearRegression model
    bridge = AdaptiveBridge(model)

    # Fit the model
    bridge.fit(X, y)

    # Make predictions
    predictions = bridge.predict(X)

    # Assert that predictions are of the correct shape
    assert predictions.shape == (len(X),)

# Test the feature importance score method


def test_feature_importance_score(sample_data):
    X, y = sample_data
    model = LinearRegression()
    # Create an AdaptiveBridge instance with the LinearRegression model
    bridge = AdaptiveBridge(model)

    # Fit the model
    bridge.fit(X, y)

    # Call the feature_importance_score method
    bridge.feature_importance_score(X)

    # No need to assert anything specific since it's a print statement

# Test the bridge method


def test_bridge_method(sample_data):
    X, y = sample_data
    model = LinearRegression()
    # Create an AdaptiveBridge instance with the LinearRegression model
    bridge = AdaptiveBridge(model)

    # Fit the model
    bridge.fit(X, y)

    # Call the bridge method
    bridged_X = bridge.bridge(X)

    # Assert that the output has the correct shape
    assert bridged_X.shape == X.shape
