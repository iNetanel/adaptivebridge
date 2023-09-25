#!/bin/env python
# featurebridge/utils/_data_validation.py
"""
    Package Name: FeatureBridge
    Author: Netanel Eliav
    Author Email: inetanel@me.com
    License: MIT License
    Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from scipy import sparse


def _convert_to_dataframe(obj, data_type):
    # Determine the type of the input object
    if isinstance(obj, np.ndarray):
        # If it's a NumPy array, convert it to a Pandas DataFrame or Pandas Series
        if data_type == "dataframe":
            return pd.DataFrame(obj)
        else:
            return pd.Series(obj)
    elif isinstance(obj, pd.DataFrame):
        # If it's already a DataFrame, return it as if it's for features and return it as is if it's for target
        if data_type == "dataframe":
            return obj
        else:
            return pd.Series(obj)
    elif isinstance(obj, pd.Series):
        # If it's already a DataFrame, return it as if it's for features and return it as is if it's for target
        if data_type == "dataframe":
            return obj.to_frame()
        else:
            return obj
    elif isinstance(obj, list):
        # If it's a Python list, convert it to a DataFrame if it's for features and return it as is if it's for target
        if data_type == "dataframe":
            return pd.DataFrame(obj)
        else:
            return pd.Series(obj)
    elif isinstance(obj, dict):
        # If it's a dictionary, convert it to a DataFrame using DictVectorizer
        if data_type == "dataframe":
            from sklearn.feature_extraction import DictVectorizer

            vec = DictVectorizer(sparse=False)
            return pd.DataFrame(vec.fit_transform(obj))
        else:
            from sklearn.feature_extraction import DictVectorizer

            vec = DictVectorizer(sparse=False)
            return pd.Series(vec.fit_transform(obj))
    elif isinstance(obj, (sparse.csr.csr_matrix, sparse.csc.csc_matrix)):
        # If it's a sparse matrix, convert it to a Pandas DataFrame
        if data_type == "dataframe":
            return pd.DataFrame(obj.toarray())
        else:
            return pd.Series(obj.toarray())
    else:
        raise ValueError(
            f"Unsupported data type {type(obj)}. Supported types are NumPy arrays, Pandas DataFrames, Python lists, dictionaries, and sparse matrices."
        )
