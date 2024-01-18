#!/bin/env python
# adaptivebridge/utils/_data_distribution.py

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
from scipy.stats import skew as skew
from distfit import distfit
from enum import Enum


# Class for distribution enumeration
class _DistributionType(Enum):
    # List of candidate continuous distributions to test
    """
    Distributions supported:
        - "norm",       # Normal distribution
        - "expon",      # Exponential distribution
        - "gamma",      # Gamma distribution
        - "dweibull",   # Weibull distribution
        - "lognorm",    # Log-normal distribution
        - "pareto",     # Pareto distribution
        - "t",          # Student's t distribution
        - "beta",       # Beta distribution
        - "uniform",    # Uniform distribution
        - "loggamma"    # Log-Gamma distribution
        - "genextreme"  # GEV distribution
    """

    NORM = "norm"
    EXPON = "expon"
    GAMMA = "gamma"
    DWEIBULL = "dweibull"
    LOGNORM = "lognorm"
    PARETO = "pareto"
    T = "t"
    BETA = "beta"
    UNIFORM = "uniform"
    LOGGAMMA = "loggamma"
    GENEXTREME = "genextreme"


# An helper function to return the most frequent mode
def _mode_selector(x_df):
    """
    Select the most frequent mode from a pandas Series.

    Parameters:
    - x_df (pd.Series): Input data as a pandas Series.

    Returns:
    - most_frequent_mode: The most frequent mode in the input data.
    """

    value_counts = x_df.value_counts()

    # Boolean handler
    # Check if there are two unique values with the same frequency
    if x_df.dtype == bool:
        if len(value_counts) == 2 and value_counts.iloc[0] == value_counts.iloc[1]:
            # In case of a tie, you might want to return None or make a decision
            return x_df.iloc[0]
        else:
            return value_counts.idxmax()

    modes = x_df.mode()
    # If there are no modes, return None
    if modes.empty or len(modes) > 2:
        return None

    # If there's only one mode, directly select it
    if len(modes) == 1:
        most_frequent_mode = modes.iloc[0]
    else:
        # If there are multiple modes, select the one with the maximum count
        most_frequent_mode = modes.iloc[0]

    return most_frequent_mode

# A function to detect the type of distribution (between continuous and discrete (AdaptiveBridge is set to 10 for discrete threshold)


def _high_level_distribution(x_df):
    # Calculate the range of the data
    data_range = max(x_df) - min(x_df)
    avg_distance = sum(abs(x - y) for x, y in zip(x_df,
                       x_df[1:])) / (len(x_df) - 1) if len(x_df) > 1 else 0

    # Count the number of unique values
    unique_values = len(set(x_df))

    # Determine the threshold for considering data as discrete
    discrete_threshold = 10

    # Avoid division by zero
    if avg_distance == 0:
        return "discrete"

    # Compare the ratio and unique values to the threshold
    if (data_range / avg_distance) <= discrete_threshold or unique_values <= discrete_threshold:
        return "discrete"
    return "continuous"


# Main central_tendency fitting for discrete data type
def _discrete_central_tendency(x_df):
    best_distribution = "discrete"
    central_tendency_value = None

    # Handle NaN values by dropping them before calculations
    x_df = x_df.dropna()

    unique_values, counts = np.unique(x_df, return_counts=True)
    most_common_value = unique_values[np.argmax(counts)]

    if len(unique_values) == 1:
        central_tendency = "constant"  # All values are the same

    elif len(unique_values) == len(x_df):
        central_tendency = "median"  # All unique values, likely skewed

    elif counts[unique_values == most_common_value] >= (len(x_df) / 2):
        modes = x_df.mode()
        # If there are no modes, return None
        if not (modes.empty or len(modes) > 2):
            central_tendency = "mode"  # Mode is most frequent for more than half of the data
    else:
        central_tendency = "median"  # Default to median for general cases

    if central_tendency == "constant":
        central_tendency_value = x_df.iloc[0]
    elif central_tendency == "median":
        central_tendency_value = x_df.median()
    elif central_tendency == "mode":
        central_tendency_value = _mode_selector(x_df)

    # Check if the data resembles integers and round it if so.
    if (x_df.astype(int) == x_df).all() and not (x_df.dtype == bool or x_df.dtype == np.bool_):
        central_tendency_value = round(central_tendency_value)

    return best_distribution, central_tendency_value, central_tendency


# Main central_tendency fitting for continuous data type
def _continuous_central_tendency(x_df):
    """
    Main central_tendency fitting for continuous data type.

    :param x_df: Input data as a pandas DataFrame.
    :return: Tuple containing the best distribution type, central tendency value, and central tendency method.
    """

    # Initialize variables to store goodness of fit results
    dfit = distfit(distr="popular")

    # Drop NaN values before calculations
    x_df = x_df.dropna()

    # Find best theoretical distribution for empirical x_df
    dfit.fit_transform(x_df, verbose=0)  # verbose=0

    min_index = dfit.summary["score"].idxmin()
    best_distribution_type = _DistributionType(
        dfit.summary.loc[min_index, "name"]).value
    central_tendency = _choose_central_tendency(best_distribution_type, x_df)
    central_tendency_value = _calculate_central_tendency(
        central_tendency, x_df)

    return best_distribution_type, central_tendency_value, central_tendency


def _choose_central_tendency(distribution_type, x_df):
    """
    Choose the appropriate central tendency method based on the distribution type.

    :param distribution_type: Type of distribution.
    :param x_df: Input data as a pandas DataFrame.
    :return: String indicating the central tendency method.
    """
    modes = x_df.mode()

    if distribution_type in ["norm", "t", "lognorm", "uniform", "loggamma", "genextreme"]:
        skewness = skew(x_df)
        if skewness > 0.5:
            return "median"
        elif skewness < -0.5:
            return "median"
        elif not (modes.empty or len(modes) > 2):
            return "mode"
        else:
            return "mean"
    elif distribution_type in ["expon", "gamma", "dweibull", "pareto", "beta"]:
        return "median"
    else:
        return "mean"


def _calculate_central_tendency(central_tendency, x_df):
    """
    Calculate the central tendency value based on the chosen method.

    :param central_tendency: String indicating the central tendency method.
    :param x_df: Input data as a pandas DataFrame.
    :return: Calculated central tendency value.
    """

    if central_tendency == "mean":
        return x_df.mean()
    elif central_tendency == "median":
        return x_df.median()
    elif central_tendency == "mode":
        return _mode_selector(x_df)
    else:
        # Return something if nothing is been return
        return x_df.iloc[0]


# Main distribution fitting method
def _fit_distribution(x_df):
    # High level distribution test
    if _high_level_distribution(x_df) == "discrete":
        (
            best_distribution,
            central_tendency_value,
            central_tendency,
        ) = _discrete_central_tendency(x_df)
    else:
        (
            best_distribution,
            central_tendency_value,
            central_tendency,
        ) = _continuous_central_tendency(x_df)

    feature_distribution = [best_distribution,
                            central_tendency, central_tendency_value]
    return feature_distribution
