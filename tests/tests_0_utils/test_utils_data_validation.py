#!/bin/env python
# tests/tests_0_utils/test_utils_data_validation.py

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
import pytest
from adaptivebridge.utils._data_validation import _convert_to_dataframe


@pytest.fixture
def x_df_data():
    return {
        "numpy_array": np.array([[1, 2], [3, 4]]),
        "pandas_dataframe": pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        "pandas_series": pd.Series([[1, 2], [3, 4]]),
        "python_list": [[1, 2], [3, 4]],
        "python_dict": {"a": 1, "b": 2},
        "sparse_matrix": sparse.csr_matrix([[1, 0], [0, 1]])}


@pytest.fixture
def y_df_data():
    return {
        "numpy_array": np.array([1, 2, 3, 4]),
        "pandas_dataframe": pd.DataFrame({"A": [1, 2, 3, 4]}),
        "pandas_series": pd.Series([1, 2, 3, 4]),
        "python_list": [1, 2, 3, 4],
        "python_dict": {"a": 1},
        "sparse_matrix": sparse.csr_matrix([1, 2, 3, 4])}


@pytest.fixture
def y_df_data_multi_d():
    return {
        "numpy_array": np.array([[1, 2], [3, 4]]),
        "pandas_dataframe": pd.DataFrame({"A": [1, 2], "B": [3, 4]}),
        "pandas_series": pd.Series([[1, 2], [3, 4]]),
        "python_list": [[1, 2], [3, 4]],
        "python_dict": {"a": 1, "b": 2},
        "sparse_matrix": sparse.csr_matrix([[1, 0], [0, 1]])}


@pytest.mark.parametrize("source_data_type, expected_result", [
    ("numpy_array", True),
    ("pandas_dataframe", True),
    ("pandas_series", True),
    ("python_list", True),
    ("python_dict", True),
    ("sparse_matrix", True),])
def test_convert_to_dataframe(source_data_type, expected_result, x_df_data):
    result = _convert_to_dataframe(x_df_data[source_data_type], "dataframe")
    assert isinstance(result, pd.DataFrame) == expected_result


@pytest.mark.parametrize("source_data_type, expected_result", [
    ("numpy_array", True),
    ("pandas_dataframe", True),
    ("pandas_series", True),
    ("python_list", True),
    ("python_dict", True),
    ("sparse_matrix", True),])
def test_convert_to_series(source_data_type, expected_result, y_df_data):
    result = _convert_to_dataframe(y_df_data[source_data_type], "series")
    assert isinstance(result, pd.Series) == expected_result


@pytest.mark.parametrize("source_data_type", ["invalid_data_type"])
# Test unsupported data type
def test_convert_to_dataframe_unsupported_type(source_data_type):
    with pytest.raises(ValueError, match="Unsupported data type"):
        _convert_to_dataframe(source_data_type, "dataframe")
    with pytest.raises(ValueError, match="Unsupported data type"):
        _convert_to_dataframe(source_data_type, "series")


@pytest.mark.parametrize("source_data_type", [
    "numpy_array", "pandas_dataframe",
    "python_list", "python_dict", "sparse_matrix"])
# Test non one-dimensional data type for series
def test_convert_to_dataframe_unsupported_dimensions(y_df_data_multi_d, source_data_type):
    with pytest.raises(ValueError, match="Y vector should be one-dimensional"):
        _convert_to_dataframe(y_df_data_multi_d[source_data_type], "series")


if __name__ == "__main__":
    pytest.main()
