#!/bin/env python
# adaptivebridge/utils/_data_distribution.py
"""
    Package Name: AdaptiveBridge
    Author: Netanel Eliav
    Author Email: inetanel@me.com
    License: MIT License
    Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import numpy as np
from scipy.stats import genextreme, gamma, lognorm, dweibull, pareto, beta
from distfit import distfit


# Define a function to choose the best central tendency method for Pareto distribution
def _pareto_choose_central_tendency(x_df):
    # Calculate the mean, median, and mode of the data based on the Pareto distribution
    shape, _, scale = pareto.fit(x_df, floc=0)

    # Calculate the mean, median, and mode based on Pareto parameters
    mean_pareto = scale / (shape - 1) if shape > 1 else np.inf
    median_pareto = scale * (2**shape - 1)
    mode_pareto = scale if shape > 1 else np.nan

    if np.isnan(mode_pareto):
        # Data may not have a well-defined mode for certain parameter so use mean or median
        return "mean" if mean_pareto <= median_pareto else "median"
    # Choose the method based on the mode and other central tendency measures
    if mean_pareto <= median_pareto:
        return "mean"
    if mode_pareto < mean_pareto:
        return "median"
    return "mode"


# Define a function to choose the best central tendency method for Double Weibull distribution
def _dweibull_choose_central_tendency(x_df):
    # Calculate the mean, median, and mode of the data based on the Double Weibull distribution
    shape, loc, scale = dweibull.fit(x_df, floc=0)
    # Mean of Double Weibull distribution
    mean_dweibull = scale * np.exp(np.log(2) / shape)
    median_dweibull = scale * (np.log(2)) ** (
        1 / shape
    )  # Median of Double Weibull distribution
    mode_dweibull = loc if shape > 1 else np.nan

    if np.isnan(mode_dweibull):
        # Data may not have a well-defined mode for certain parameter so use mean or median
        return "mean" if mean_dweibull <= median_dweibull else "median"
    # Choose the method based on the mode and other central tendency measures
    if mean_dweibull <= median_dweibull:
        return "mean"
    if mode_dweibull < mean_dweibull:
        return "median"
    return "mode"


# Define a function to choose the best central tendency method for GEV distribution
def _genextreme_choose_central_tendency(x_df):
    # Calculate the mean, median, and mode of the data based on the GEV distribution
    shape, loc, scale = genextreme.fit(x_df)
    mean_gev = loc + scale * (1 - shape) ** (-1) if shape != 0 else np.inf
    median_gev = (
        loc + scale * (np.log(2) ** (-shape) - 1) / shape
        if shape != 0
        else loc - scale * np.log(np.log(2))
    )
    mode_gev = loc + scale * (1 - shape) ** (-1) if shape < 1 else np.nan

    if np.isnan(mode_gev):
        # Data may not have a well-defined mode for certain parameter so use mean or median
        return "mean" if mean_gev <= median_gev else "median"
    # Choose the method based on the mode and other central tendency measures
    if mean_gev <= median_gev:
        return "mean"
    if mode_gev < mean_gev:
        return "median"
    return "mode"


# Define a function to choose the best central tendency method for Gamma distribution
def _gamma_choose_central_tendency(x_df):
    # Calculate the mean, median, and mode of the data based on the gamma distribution
    shape, loc, scale = gamma.fit(x_df)
    mean_gamma = shape * scale
    median_gamma = gamma.ppf(0.5, shape, loc=loc, scale=scale)
    mode_gamma = (shape - 1) * scale  # Mode for shape > 1

    if shape <= 1:
        # Data may not have a well-defined mode for shape <= 1, so use the mean or median
        return "mean" if mean_gamma <= median_gamma else "median"
    # Choose the method based on the shape parameter
    if mean_gamma <= median_gamma:
        return "mean"
    if mode_gamma < mean_gamma:
        return "median"
    return "mode"


def _lognorm_choose_central_tendency(x_df):
    # Calculate the mean and median of the log-transformed data for Log-Transformed distribution
    log_data = np.log(x_df)
    mean_log = np.mean(log_data)
    # median_log = np.median(log_data) optional for future use

    # Calculate the skewness of the log-transformed data
    log_data_skewness = lognorm.stats(
        s=1, loc=mean_log, scale=np.exp(mean_log), moments="s"
    )

    if abs(log_data_skewness) < 0.5:
        # Log-transformed data is approximately symmetric
        return "mean"
    if log_data_skewness < -0.5:
        # Log-transformed data is left-skewed
        return "median"
    if log_data_skewness > 0.5:
        # Log-transformed data is right-skewed
        return "median"
    # Default to the mode
    return "mode"


# Define a function to choose the best central tendency method for Beta distribution
def _beta_choose_central_tendency(x_df):
    # Calculate the mean, median, and mode of the data based on the beta distribution
    distribution_alpha, distribution_beta, loc, scale = beta.fit(x_df, floc=0)

    # Calculate the mean, median, and mode based on beta parameters
    mean_beta = (
        distribution_alpha / (distribution_alpha + distribution_beta)
    ) * scale + loc
    median_beta = beta.ppf(
        0.5, distribution_alpha, distribution_beta, loc=loc, scale=scale
    )
    if distribution_alpha > 1 and distribution_beta > 1:
        mode_beta = (distribution_alpha - 1) / (
            distribution_alpha + distribution_beta - 2
        ) * scale + loc
    else:
        mode_beta = np.nan

    if np.isnan(mode_beta):
        # Data may not have a well-defined mode for certain parameter so use mean or median
        return "mean" if mean_beta <= median_beta else "median"

    # Choose the method based on the mode and other central tendency measures
    if mean_beta <= median_beta:
        return "mean"
    if mode_beta < mean_beta:
        return "median"
    return "mode"


# Define a function to choose the best central tendency method for Uniform distribution
def _uniform_choose_central_tendency(x_df):
    mean_uniform = np.mean(x_df)
    median_uniform = np.median(x_df)

    # You can choose between mean and median based on your preference
    if mean_uniform <= median_uniform:
        return "mean"
    return "median"


# Define a function to choose the best central tendency method for Log-Gamma distribution
def _loggamma_choose_central_tendency(_):
    # TODO > technical debt for Log-Gamma distribution.
    return "mean"


# Define a function to detect the type of distribution (between continuous and discrete)
def _high_level_distribution(x_df):
    # Calculate the range of the data
    data_range = max(x_df) - min(x_df)

    # Count the number of unique values
    unique_values = len(set(x_df))

    # Determine the threshold for considering data as discrete
    discrete_threshold = 10

    # Compare range and unique values to the threshold
    if data_range <= discrete_threshold or unique_values <= discrete_threshold:
        return "discrete"
    return "continuous"


# Main central_tendency fitting for discrete data type


def _discrete_central_tendency_set(x_df):
    best_distribution = "discrete"
    unique_values, counts = np.unique(x_df, return_counts=True)
    most_common_value = unique_values[np.argmax(counts)]

    if len(unique_values) == 1:
        central_tendency = "constant"  # All values are the same

    if len(unique_values) == len(x_df):
        central_tendency = "median"  # All unique values, likely skewed

    if counts[unique_values == most_common_value] >= (len(x_df) / 2):
        central_tendency = (
            "mode"  # Mode is most frequent for more than half of the data
        )

    central_tendency = "median"  # Default to median for general cases

    if central_tendency == "mean":
        central_tendency_value = x_df.mean()
    elif central_tendency == "median":
        central_tendency_value = x_df.median()
    elif central_tendency == "mode":
        mode_values = x_df.mode()
        central_tendency_value = mode_values.iloc[0]
    else:
        central_tendency_value = x_df.iloc[0]

    # NaN fix in case that count for True and False are the same
    if np.isnan(central_tendency_value):
        if x_df.dtype == bool:
            central_tendency_value = True
        else:
            central_tendency_value = 1

    # Check if the data resembles integers and round it if so.
    is_integer_column = x_df.apply(
        lambda x: int(x) == x if str(x).replace(".", "").isdigit() else False
    )

    if is_integer_column.all():
        central_tendency_value = round(central_tendency_value)
    return best_distribution, central_tendency_value, central_tendency


# Main central_tendency fitting for continuous data type
def _continuous_central_tendency_set(x_df):
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
    """

    # Initialize variables to store goodness of fit results
    dfit = distfit(distr="popular")

    # Find best theoretical distribution for empirical x_df
    dfit.fit_transform(x_df, verbose=0)  # verbose=0

    min_index = dfit.summary["score"].idxmin()
    best_distribution = dfit.summary.loc[min_index, "name"]

    if best_distribution == "norm":
        central_tendency = "mean"
    elif best_distribution == "expon":
        central_tendency = "median"
    elif best_distribution == "pareto":
        central_tendency = _pareto_choose_central_tendency(x_df)
    elif best_distribution == "dweibull":
        central_tendency = _dweibull_choose_central_tendency(x_df)
    elif best_distribution == "t":
        central_tendency = "mean"
    elif best_distribution == "genextreme":
        central_tendency = _genextreme_choose_central_tendency(x_df)
    elif best_distribution == "gamma":
        central_tendency = _gamma_choose_central_tendency(x_df)
    elif best_distribution == "lognorm":
        central_tendency = _lognorm_choose_central_tendency(x_df)
    elif best_distribution == "beta":
        central_tendency = _beta_choose_central_tendency(x_df)
    elif best_distribution == "uniform":
        central_tendency = _uniform_choose_central_tendency(x_df)
    elif best_distribution == "loggamma":
        central_tendency = _loggamma_choose_central_tendency(x_df)

    if central_tendency == "mean":
        central_tendency_value = x_df.mean()
    elif central_tendency == "median":
        central_tendency_value = x_df.median()
    elif central_tendency == "mode":
        mode_values = x_df.mode()
        central_tendency_value = mode_values.iloc[0]
    else:
        central_tendency_value = x_df.iloc[0]

    return best_distribution, central_tendency_value, central_tendency


# Main distribution fitting method


def _fit_distribution(x_df):
    # High level distribution test
    if _high_level_distribution(x_df) == "discrete":
        (
            best_distribution,
            central_tendency_value,
            central_tendency,
        ) = _discrete_central_tendency_set(x_df)
    else:
        (
            best_distribution,
            central_tendency_value,
            central_tendency,
        ) = _continuous_central_tendency_set(x_df)

    feature_distribution = [best_distribution,
                            central_tendency, central_tendency_value]
    return feature_distribution
