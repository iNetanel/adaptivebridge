#!/bin/env python
# featurebridge/utils/_data_distribution.py
"""
    Package Name: FeatureBridge
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
    shape, loc, scale = pareto.fit(x_df, floc=0)

    # Calculate the mean, median, and mode based on Pareto parameters
    mean_pareto = scale / (shape - 1) if shape > 1 else np.inf
    median_pareto = scale * (2**shape - 1)
    mode_pareto = scale if shape > 1 else np.nan

    if np.isnan(mode_pareto):
        # Data may not have a well-defined mode for certain parameter combinations, so use mean or median
        return "mean" if mean_pareto <= median_pareto else "median"
    else:
        # Choose the method based on the mode and other central tendency measures
        if mean_pareto <= median_pareto:
            return "mean"
        elif mode_pareto < mean_pareto:
            return "median"
        else:
            return "mode"


# Define a function to choose the best central tendency method for Double Weibull distribution
def _dweibull_choose_central_tendency(x_df):
    # Calculate the mean, median, and mode of the data based on the Double Weibull distribution
    c, loc, scale = dweibull.fit(x_df, floc=0)
    mean_dweibull = scale * np.exp(np.log(2) / c)  # Mean of Double Weibull distribution
    median_dweibull = scale * (np.log(2)) ** (
        1 / c
    )  # Median of Double Weibull distribution
    mode_dweibull = loc if c > 1 else np.nan

    if np.isnan(mode_dweibull):
        # Data may not have a well-defined mode for certain parameter combinations, so use mean or median
        return "mean" if mean_dweibull <= median_dweibull else "median"
    else:
        # Choose the method based on the mode and other central tendency measures
        if mean_dweibull <= median_dweibull:
            return "mean"
        elif mode_dweibull < mean_dweibull:
            return "median"
        else:
            return "mode"


# Define a function to choose the best central tendency method for GEV distribution
def _genextreme_choose_central_tendency(x_df):
    # Calculate the mean, median, and mode of the data based on the GEV distribution
    c, loc, scale = genextreme.fit(x_df)
    mean_gev = loc + scale * (1 - c) ** (-1) if c != 0 else np.inf
    median_gev = (
        loc + scale * (np.log(2) ** (-c) - 1) / c
        if c != 0
        else loc - scale * np.log(np.log(2))
    )
    mode_gev = loc + scale * (1 - c) ** (-1) if c < 1 else np.nan

    if np.isnan(mode_gev):
        # Data may not have a well-defined mode for certain parameter combinations, so use mean or median
        return "mean" if mean_gev <= median_gev else "median"
    else:
        # Choose the method based on the mode and other central tendency measures
        if mean_gev <= median_gev:
            return "mean"
        elif mode_gev < mean_gev:
            return "median"
        else:
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
    else:
        # Choose the method based on the shape parameter
        if mean_gamma <= median_gamma:
            return "mean"
        elif mode_gamma < mean_gamma:
            return "median"
        else:
            return "mode"


def _lognorm_choose_central_tendency(x_df):
    # Calculate the mean and median of the log-transformed data for Log-Transformed distribution
    log_data = np.log(x_df)
    mean_log = np.mean(log_data)
    median_log = np.median(log_data)

    # Calculate the skewness of the log-transformed data
    log_data_skewness = lognorm.stats(
        s=1, loc=mean_log, scale=np.exp(mean_log), moments="s"
    )

    if abs(log_data_skewness) < 0.5:
        # Log-transformed data is approximately symmetric
        return "mean"
    elif log_data_skewness < -0.5:
        # Log-transformed data is left-skewed
        return "median"
    elif log_data_skewness > 0.5:
        # Log-transformed data is right-skewed
        return "median"
    else:
        # Default to the mode
        return "mode"


# Define a function to choose the best central tendency method for Beta distribution
def _beta_choose_central_tendency(x_df):
    # Calculate the mean, median, and mode of the data based on the beta distribution
    a, b, loc, scale = beta.fit(x_df, floc=0)

    # Calculate the mean, median, and mode based on beta parameters
    mean_beta = (a / (a + b)) * scale + loc
    median_beta = beta.ppf(0.5, a, b, loc=loc, scale=scale)
    if a > 1 and b > 1:
        mode_beta = (a - 1) / (a + b - 2) * scale + loc
    else:
        mode_beta = np.nan

    if np.isnan(mode_beta):
        # Data may not have a well-defined mode for certain parameter combinations, so use mean or median
        return "mean" if mean_beta <= median_beta else "median"
    else:
        # Choose the method based on the mode and other central tendency measures
        if mean_beta <= median_beta:
            return "mean"
        elif mode_beta < mean_beta:
            return "median"
        else:
            return "mode"


# Define a function to choose the best central tendency method for Uniform distribution
def _uniform_choose_central_tendency(x_df):
    mean_uniform = np.mean(x_df)
    median_uniform = np.median(x_df)

    # You can choose between mean and median based on your preference
    if mean_uniform <= median_uniform:
        return "mean"
    else:
        return "median"


# Define a function to choose the best central tendency method for Log-Gamma distribution
def _loggamma_choose_central_tendency(x_df):
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
    else:
        return "continuous"


# Main distribution fitting method
def _fit_distribution(x_df):
    # High level distribution test
    if _high_level_distribution(x_df) == "discrete":
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

    else:
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

    feature_distribution = [best_distribution, central_tendency, central_tendency_value]
    return feature_distribution
