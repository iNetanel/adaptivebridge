#!/bin/env python
'''
    Project Name: FeatureBridge
    Author: Netanel Eliav
    Author Email: inetanel@me.com
    License: MIT License
    Version: Please refer to the repository for the latest version and updates.
'''

# Import necessary libraries
import numpy as np
from scipy.stats import genextreme, gamma, lognorm, dweibull, pareto, beta
from distfit import distfit

class Utils():
    def __init__(self):
        pass

    # Define a function to choose the best central tendency method for Pareto distribution
    def pareto_choose_central_tendency(self, x_df):
        # Calculate the mean, median, and mode of the data based on the Pareto distribution
        shape, loc, scale = pareto.fit(x_df, floc=0)
        
        # Calculate the mean, median, and mode based on Pareto parameters
        mean_pareto = scale / (shape - 1) if shape > 1 else np.inf
        median_pareto = scale * (2**shape - 1)
        mode_pareto = scale if shape > 1 else np.nan

        if np.isnan(mode_pareto):
            # Data may not have a well-defined mode for certain parameter combinations, so use mean or median
            return "Mean" if mean_pareto <= median_pareto else "median"
        else:
            # Choose the method based on the mode and other central tendency measures
            if mean_pareto <= median_pareto:
                return "mean"
            elif mode_pareto < mean_pareto:
                return "median"
            else:
                return "mode"

    # Define a function to choose the best central tendency method for Double Weibull distribution
    def dweibull_choose_central_tendency(self, x_df):
        # Calculate the mean, median, and mode of the data based on the Double Weibull distribution
        c, loc, scale = dweibull.fit(x_df, floc=0)
        mean_dweibull = scale * np.exp(np.log(2) / c)  # Mean of Double Weibull distribution
        median_dweibull = scale * (np.log(2))**(1/c)  # Median of Double Weibull distribution
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
    def genextreme_choose_central_tendency(self, x_df):
        # Calculate the mean, median, and mode of the data based on the GEV distribution
        c, loc, scale = genextreme.fit(x_df)
        mean_gev = loc + scale * (1 - c)**(-1) if c != 0 else np.inf
        median_gev = loc + scale * (np.log(2)**(-c) - 1) / c if c != 0 else loc - scale * np.log(np.log(2))
        mode_gev = loc + scale * (1 - c)**(-1) if c < 1 else np.nan

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
    def gamma_choose_central_tendency(self, x_df):
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

    def lognorm_choose_central_tendency(self, x_df):
        # Calculate the mean and median of the log-transformed data for Log-Transformed distribution
        log_data = np.log(x_df)
        mean_log = np.mean(log_data)
        median_log = np.median(log_data)

        # Calculate the skewness of the log-transformed data
        log_data_skewness = lognorm.stats(s=1, loc=mean_log, scale=np.exp(mean_log), moments='s')

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
    def beta_choose_central_tendency(self, x_df):
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
    def uniform_choose_central_tendency(self, x_df):
        mean_uniform = np.mean(x_df)
        median_uniform = np.median(x_df)
        
        # You can choose between mean and median based on your preference
        if mean_uniform <= median_uniform:
            return "mean"
        else:
            return "median"

    # Define a function to choose the best central tendency method for Log-Gamma distribution
    def loggamma_choose_central_tendency(self, x_df):
        # TODO > technical debt for Log-Gamma distribution.
        return "mean"

    # Main distribution fitting method
    def fit_distribution(self, x_df):
        # List of candidate distributions to test
        '''
            Distributions supported:
                - "norm",        # Normal distribution
                - "expon",       # Exponential distribution
                - "gamma",       # Gamma distribution
                - "dweibull", # Weibull distribution
                - "lognorm",     # Log-normal distribution
                - "pareto",      # Pareto distribution
                - "t",            # Student's t distribution
                - "beta",         # Beta distribution
                - "uniform",      # Uniform distribution
                - "loggamma"      # Log-Gamma distribution
        '''

        # Initialize variables to store goodness of fit results
        dfit = distfit(distr='popular')
        
        # Find best theoretical distribution for empirical x_df
        dfit.fit_transform(x_df, verbose=0) #verbose=0
        
        min_index = dfit.summary['score'].idxmin()
        best_distribution = dfit.summary.loc[min_index, 'name']

        if best_distribution == 'norm':
            central_tendency = 'mean'
        elif best_distribution == 'expon':
            central_tendency = 'median'
        elif best_distribution == 'pareto':
                central_tendency = self.pareto_choose_central_tendency(x_df)
        elif best_distribution == 'dweibull':
                central_tendency = self.dweibull_choose_central_tendency(x_df)
        elif best_distribution == 't':
            central_tendency = 'mean'
        elif best_distribution == 'genextreme':
            central_tendency = self.genextreme_choose_central_tendency(x_df)
        elif best_distribution == 'gamma':
            central_tendency = self.gamma_choose_central_tendency(x_df)
        elif best_distribution == 'lognorm':
            central_tendency = self.lognorm_choose_central_tendency(x_df)
        elif best_distribution == 'beta':
            central_tendency = self.beta_choose_central_tendency(x_df)
        elif best_distribution == 'uniform':
            central_tendency = self.uniform_choose_central_tendency(x_df)
        elif best_distribution == 'loggamma':
            central_tendency = self.loggamma_choose_central_tendency(x_df)

        if central_tendency == 'mean':
            central_tendency_value = x_df.mean()
        elif central_tendency == 'median':
            central_tendency_value = x_df.median()
        else:
            mode_values = x_df.mode()
            central_tendency_value = mode_values.iloc[0]
        feature_distribution = [best_distribution, central_tendency, central_tendency_value]
        return feature_distribution