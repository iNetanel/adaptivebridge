#!/bin/env python
# adaptivebridge/utils/_data_validation.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
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
        if obj.ndim == 1:
            return pd.Series(obj)
        else:
            raise ValueError(
                f"Y vector should be one-dimensional and it's {obj.ndim} dimensions.")
    if isinstance(obj, pd.DataFrame):
        # Return a DataFrame for features, and Series for target
        if data_type == "dataframe":
            return obj
        if obj.shape[1] == 1:
            return obj.iloc[:, 0]
        else:
            raise ValueError(
                f"Y vector should be one-dimensional and it's {obj.shape[1]} dimensions.")
    if isinstance(obj, pd.Series):
        # Return a DataFrame for features, and Series for target
        if data_type == "dataframe":
            return obj.to_frame()
        return obj
    if isinstance(obj, list):
        # If it's a Python list, convert it to a DataFrame for features and Series for target
        if data_type == "dataframe":
            return pd.DataFrame(obj)
        if not any(isinstance(item, (list, dict)) for item in obj):
            return pd.Series(obj)
        else:
            dimensions = sum(isinstance(item, (list, dict)) for item in obj)
            raise ValueError(
                f"Y vector should be one-dimensional and it's {dimensions} dimensions.")
    if isinstance(obj, dict):
        # If it's a dictionary, convert it to a DataFrame using DictVectorizer
        vec = DictVectorizer(sparse=False)
        if data_type == "dataframe":
            return pd.DataFrame(vec.fit_transform(obj))
        if len(obj) == 1:
            return pd.Series(vec.fit_transform(obj)[0])
        else:
            raise ValueError(
                f"Y vector should be one-dimensional and it's {len(obj)} dimensions.")
    if isinstance(obj, (sparse.csr_matrix, sparse.csc_matrix)):
        # If it's a sparse matrix, convert it to a Pandas DataFrame
        if data_type == "dataframe":
            return pd.DataFrame(obj.toarray())
        if obj.shape[0] == 1:
            return pd.Series(obj.toarray()[0])
        else:
            raise ValueError(
                f"Y vector should be one-dimensional and it's {obj.shape[0]} dimensions.")
    raise ValueError(
        f"Unsupported data type {type(obj)}. Supported types are NumPy arrays,"
        " Pandas DataFrames, Python lists, Dictionaries, and Sparse Matrices."
    )
