#!/bin/env python
# adaptivebridge/utils/_data_validation.py
"""
    Package Name: AdaptiveBridge
    Author: Netanel Eliav
    Author Email: inetanel@me.com
    License: MIT License
    Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer


def _convert_to_dataframe(obj, data_type):
    # Determine the type of the input object
    if isinstance(obj, np.ndarray):
        # If it's a NumPy array, convert it to a Pandas DataFrame or Pandas Series
        if data_type == "dataframe":
            return pd.DataFrame(obj)
        return pd.Series(obj)
    if isinstance(obj, pd.DataFrame):
        # Return a DataFrame for features, and Series for target
        if data_type == "dataframe":
            return obj
        return pd.Series(obj)
    if isinstance(obj, pd.Series):
        # Return a DataFrame for features, and Series for target
        if data_type == "dataframe":
            return obj.to_frame()
        return obj
    if isinstance(obj, list):
        # If it's a Python list, convert it to a DataFrame for features and Series for target
        if data_type == "dataframe":
            return pd.DataFrame(obj)
        return pd.Series(obj)
    if isinstance(obj, dict):
        # If it's a dictionary, convert it to a DataFrame using DictVectorizer
        if data_type == "dataframe":
            vec = DictVectorizer(sparse=False)
            return pd.DataFrame(vec.fit_transform(obj))
        vec = DictVectorizer(sparse=False)
        return pd.Series(vec.fit_transform(obj))
    if isinstance(obj, (sparse.csr.csr_matrix, sparse.csc.csc_matrix)):
        # If it's a sparse matrix, convert it to a Pandas DataFrame
        if data_type == "dataframe":
            return pd.DataFrame(obj.toarray())
        return pd.Series(obj.toarray())
    raise ValueError(
        f"Unsupported data type {type(obj)}. Supported types are NumPy arrays,"
        " Pandas DataFrames, Python lists, dictionaries, and sparse matrices."
    )
