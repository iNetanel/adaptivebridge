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
import pandas as pd
import copy
import matplotlib.pyplot as plt
from itertools import combinations

# Define a class named FeatureBridge
class FeatureBridge:
    def __init__(self, model, correlation_threshold=0.3, min_accuracy=0.5, default_accuracy_selection=0.95, importancy_threshold=0.1, accuracy_logic=None):
        """
        Initialize the main FeatureBridge object.

        Parameters:
            model (object): The machine learning model (e.g., LinearRegression) to be used for modeling.
            correlation_threshold (float): The correlation threshold for feature selection based on correlation.
            min_accuracy (float): The minimum accuracy required for feature selection.
            default_accuracy_selection (float): The default accuracy threshold for feature selection.
            importancy_threshold (float): The threshold for feature importance.
            accuracy_logic (function): Custom accuracy calculation logic (optional).

        Returns:
            None
        """

        self.correlation_threshold = correlation_threshold
        self.min_accuracy = min_accuracy
        self.importancy_th = importancy_threshold
        self.default_accuracy_selection = default_accuracy_selection
        self.model = copy.deepcopy(model)  # Create a deep copy of the provided machine learning model
        self.accuracy_logic = accuracy_logic
        self.x_df = None  # Placeholder for feature data frame
        self.y_df = None  # Placeholder for target data frame
        self.im = None  # Placeholder for basic feature importance
        self.corr_matrix = None  # Placeholder for the correlation matrix
        self.max_feature = None  # Placeholder for the most important feature
        self.max_index = None  # Placeholder for the index of the maximum feature importance
        self.data_distribution = None  # Placeholder for data distribution statistics
        self.feature_distribution = None  # Placeholder for feature distribution statistics
        self.model_map = None  # Placeholder for a mapping of features and models

    # Define a custom exception class
    class MandatoryFeatureError(Exception):
        pass
    
    # Define a string representation for the class
    def __str__(self):
        """
        Define the string representation of the object.

        Returns:
            FeatureBridge internal information
        """
    
        return f'{self.x_df}({self.im})'

    # Method to fit the model to the data
    def fit(self, x_df, y_df):
        """
        Fit the machine learning model to the input data.

        Parameters:
            x_df (DataFrame): The feature data frame.
            y_df (Series or array-like): The target variable data.

        Returns:
            None
        """

        self.model = self.model.fit(x_df, y_df)
        self.x_df = x_df  # Assign the feature data frame
        self.y_df = y_df  # Assign the target data frame
        self.im = self.calculate_importance()  # Calculate basic feature importance
        self.corr_matrix = self.x_df.corr()  # Calculate the correlation matrix
        self.max_feature = self.im.max()  # Find the most important feature
        self.max_index = self.im.idxmax()  # Find the index of the maximum feature importance
        self.data_distribution = self.x_df.describe()  # Calculate data distribution statistics
        self.feature_distribution = self.distribution()  # Calculate feature distribution statistics
        self.model_map = self.model_mapping()  # Create a mapping of features and models
        self.feature_mapping()

    # Method to make predictions
    def predict(self, x_df):
        """
        Make predictions using the fitted model.

        Parameters:
            x_df (DataFrame): The feature data frame for prediction.

        Returns:
            array: Predicted values.
        """

        x_df = self.df_fit(x_df)
        return self.model.predict(x_df)

    # Method to calculate feature importance
    def calculate_importance(self):
        """
        Calculate the basic feature importance.

        Returns:
            Series: Feature importances.
        """

        return np.abs(self.model.coef_ * self.x_df.mean())

    # Method to print feature importance scores
    def score(self):
        """
        Summarize feature importance scores.

        Returns:
            None
        """

        for i, v in enumerate(self.im):
            print('Feature: %0d, Score: %.5f' % (i, v))

    # Method to determine feature distribution characteristics
    def distribution(self):
        """
        Determine the method for handling feature distribution based on data characteristics.

        Returns:
            dict: Mapping of features to distribution methods.
        """

        feature_distribution = {}
        for feature in self.x_df.columns:
            # Calculate coefficient of variation (CV)
            cv = self.data_distribution.loc['std', feature] / self.data_distribution.loc['mean', feature]
            if cv >= 1:
                cv = 'mean'
            else:
                cv = '50%'
            # Check if data is binary (0 or 1)
            unique_values = self.x_df[feature].unique()
            if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
                if self.data_distribution.loc['mean', feature] > 0.5:
                    feature_distribution[feature] = 1
                else:
                    feature_distribution[feature] = 0
            # Check if data is boolean (True or False)
            elif self.x_df[feature].dtype == bool:
                value_counts = self.x_df[feature].value_counts()
                if value_counts[True] > value_counts[False]:
                    feature_distribution[feature] = True
                else:
                    feature_distribution[feature] = False
            else:
                # Check if the data resembles integers
                is_integer_column = self.x_df[feature].apply(lambda x: int(x) == x if str(x).replace(".", "").isdigit() else False)

                if is_integer_column.all():
                    feature_distribution[feature] = round(self.data_distribution.loc[cv, feature])
                else:
                    feature_distribution[feature] = self.data_distribution.loc[cv, feature]

        return feature_distribution

    # Method to create a mapping of features
    def mapping(self):
        """
        Create a mapping of features.

        Returns:
            dict: Mapping of features to data frames.
        """

        df_map = {}
        for column in self.x_df.columns:
            df_map[column] = self.x_df[column]
        return df_map

    # Method to identify features to drop based on correlation
    def drop_matrix(self, feature):
        """
        Identify features to drop based on correlation with the specified feature.

        Parameters:
            feature (str): The target feature.

        Returns:
            list: List of features to drop.
        """

        matrix = np.abs(self.corr_matrix)
        criteria1 = matrix < self.correlation_threshold
        criteria2 = matrix == 1
        cleared_matrix = matrix[criteria1 | criteria2]
        drop_matrix = cleared_matrix[feature]
        drop_matrix = drop_matrix.dropna()
        return drop_matrix.index.tolist()

    # Method to generate all combinations of features
    def all_combinations(self, x_df):
        """
        Generate all possible combinations of features.

        Parameters:
            x_df (DataFrame): The feature data frame.

        Returns:
            list: List of feature combinations.
        """

        all_combinations = []
        for r in range(1, len(x_df) + 1):
            comb_r = list(combinations(x_df, r))
            all_combinations.extend(comb_r)
        return all_combinations

    # Method to create a mapping of models for each feature
    def model_mapping(self):
        """
        Map models to features based on accuracy.

        Returns:
            dict: Mapping of features to models and accuracy.
        """

        model = copy.deepcopy(self.model)
        model_map = {}
        for feature in self.x_df.columns:
            x_df = self.x_df.drop(self.drop_matrix(feature), axis=1)
            y_df = self.x_df[feature].values.reshape(-1, 1)
            i = 0
            if len(x_df.columns) == 0:
                model_map[feature] = {i: {'accuracy': None, 'distribution': self.feature_distribution[feature], 'features': None, 'model': None}}
                continue
            model_map[feature] = {i: {'accuracy': None, 'distribution': self.feature_distribution[feature], 'features': None, 'model': None}}
            combinations = self.all_combinations(x_df)
            for combination in combinations:
                combination = list(combination)
                if len(combination) != len(x_df.columns):
                    x_df_droped = x_df.drop(combination, axis=1)
                else:
                    break
                model.fit(x_df_droped, y_df)
                ypred = model.predict(x_df_droped)
                acc = self.accuracy(y_df, ypred)
                if acc < self.min_accuracy:
                    if len(model_map[feature]) < 1:
                        del model_map[feature][i]
                else:
                    model_map[feature][i] = {'accuracy': acc, 'distribution': self.feature_distribution[feature], 'features': list(x_df_droped.columns), 'model': copy.deepcopy(model)}
                    if acc >= self.default_accuracy_selection:
                        break
                i += 1
        return model_map

    # Method to identify mandatory and deviation features
    def mandatory_and_distribution(self):
        """
        Identify mandatory and deviation features based on importance.

        Returns:
            None
        """

        for feature in self.model_map:
            for i in self.model_map[feature]:
                if self.model_map[feature][i]['features'] is None:
                    if (self.im[feature] / (np.sum(self.im, axis=0))) > self.importancy_th:
                        self.feature_map['mandatory'][feature] = self.model_map[feature][i]
                    else:
                        self.feature_map['deviation'][feature] = self.model_map[feature][i]

        for feature in self.feature_map['mandatory']:
            del self.model_map[feature]
        for feature in self.feature_map['deviation']:
            del self.model_map[feature]

    # Method to perform feature leveling
    def feature_leveling(self):
        """
        Perform feature leveling based on dependencies.

        Returns:
            None
        """

        while len(self.model_map) > 0:
            model_map_l1 = {}
            for feature in self.model_map:
                found_keys = list(self.feature_map['mandatory'].keys()) + list(self.feature_map['deviation'].keys()) + list(self.feature_map['level'].keys())
                if len(found_keys) > 0:
                    for i in self.model_map[feature]:
                        found_in_keys = [item in self.model_map[feature][i]['features'] for item in found_keys]
                        if found_in_keys and found_in_keys.count(True) == len(self.model_map[feature][i]['features']):
                            if feature in model_map_l1:
                                if model_map_l1[feature]['accuracy'] < self.model_map[feature][i]['accuracy']:
                                    model_map_l1[feature] = self.model_map[feature][i]
                            else:
                                model_map_l1[feature] = self.model_map[feature][i]
            if not model_map_l1:
                for feature in self.model_map:
                    for i in self.model_map[feature]:
                        if feature in model_map_l1:
                            if model_map_l1[feature]['accuracy'] < self.model_map[feature][i]['accuracy']:
                                model_map_l1[feature] = self.model_map[feature][i]
                        else:
                            model_map_l1[feature] = self.model_map[feature][i]

                if model_map_l1:
                    model_map_l1_sorted = dict(sorted(model_map_l1.items(), key=lambda item: item[1]['accuracy'], reverse=False))
                    if (self.im[next(iter(model_map_l1_sorted))]/(np.sum(self.im, axis=0))) > self.importancy_th:
                        self.feature_map['mandatory'][next(iter(model_map_l1_sorted))] = model_map_l1_sorted[next(iter(model_map_l1_sorted))]
                        del self.model_map[next(iter(model_map_l1_sorted))]
                    else:
                        self.feature_map['deviation'][next(iter(model_map_l1_sorted))] = model_map_l1_sorted[next(iter(model_map_l1_sorted))]
                        del self.model_map[next(iter(model_map_l1_sorted))]
                    model_map_l1 = {}
            if model_map_l1:
                model_map_l1_sorted = dict(sorted(model_map_l1.items(), key=lambda item: item[1]['accuracy'], reverse=True))
                self.feature_map['level'][next(iter(model_map_l1_sorted))] = model_map_l1_sorted[next(iter(model_map_l1_sorted))]
                del self.model_map[next(iter(model_map_l1_sorted))]

    # Method to print the feature mapping
    def dependencies(self):
        """
        Display feature dependencies and their handling methods.

        Returns:
            None
        """

        print('Feature Dependencies are:')
        print('Mandatory: (Must be provided by the user)')
        for i in self.feature_map['mandatory']:
            print(f' -{i}')
        print('\nStandard Deviation: (data distribution will be used and not prediction)')
        for i in self.feature_map['deviation']:
            print(' -{}, {}'.format(i, self.feature_map['deviation'][i]['distribution']))
        print('\nPrediction loop: (will be executed according to the dependencies)')
        for i in self.feature_map['level']:
            print(' -{}, {}'.format(i, self.feature_map['level'][i]['features']))
        print('\n')

    def feature_mapping(self):
        """
        Perform feature mapping and selection.

        Returns:
            None
        """

        self.feature_map = {'mandatory': {},
                            'deviation': {},
                            'level': {},
                            }
        self.mandatory_and_distribution()
        self.feature_leveling()

    # Method to calculate prediction accuracy
    def accuracy(self, y_df, ypred):
        """
        Calculate prediction accuracy.

        Parameters:
            y_df (Series or array-like): The true target variable values.
            ypred (array): The predicted target variable values.

        Returns:
            float: Prediction accuracy.
        """

        if self.accuracy_logic is None:
            y_sum = np.sum(y_df)
            margin = np.abs(np.subtract(ypred, y_df))
            error_sum = np.sum(margin)
            accu = 1 - (error_sum / y_sum)
            if accu < 0:
                accu = 0
        else:
            accu = self.accuracy_logic(y_df, ypred)
        return accu

    # Method to predict using the adaptive model
    def predict_adaptive(self, x_df, feature):
        """
        Make adaptive predictions for a specific feature.

        Parameters:
            x_df (DataFrame): The feature data frame.
            feature (str): The target feature.

        Returns:
            array: Predicted values for the specified feature.
        """

        if len(x_df.shape) == 1:
            x_df = x_df.values.reshape(-1, 1)
        else:
            x_df = pd.DataFrame(x_df)
        prediction = self.feature_map['level'][feature]['model'].predict(x_df)
        return prediction.flatten().astype(float)

    # Method to prepare the input data frame for prediction
    def df_fit(self, x_df):
        """
        Prepare the input data frame for prediction.

        Parameters:
            x_df (DataFrame): The feature data frame.

        Returns:
            DataFrame: Prepared feature data frame for prediction.
        """

        for feature in self.feature_map['mandatory']:
            if feature not in x_df.columns:
                raise self.MandatoryFeatureError("A mandatory feature is completely missing: {}".format(feature))
            else:
                if x_df[feature].isna().any().any():
                    raise self.MandatoryFeatureError("A mandatory feature is partially missing: {}".format(feature))

        # Handling of deviation features
        for feature in self.feature_map['deviation']:
            if feature not in x_df.columns:
                x_df[feature] = self.feature_map['deviation'][feature]['distribution']
            if x_df[feature].isna().any().any():
                x_df[feature] = x_df[feature].fillna(self.feature_map['deviation'][feature]['distribution'])

        # Handling of prediction loop features
        for feature in self.feature_map['level']:
            if feature not in x_df.columns:
                x_df[feature] = self.predict_adaptive(x_df[self.feature_map['level'][feature]['features']], feature)
            if x_df[feature].isna().any().any():
                pass  # TODO: Allow partial missing values for these features.

        # Reorder columns to match the original data frame
        x_df = x_df.reindex(columns=self.x_df.columns)
        return x_df

    # Method to benchmark the model
    def benchmark(self, x_test_df, y_text_df):
        """
        Evaluate the model's performance and impact of feature selection.

        Parameters:
            x_test_df (DataFrame): The feature data frame.
            y_text_df (DataFrame): The feature data frame.

        Returns:
            None
        """
        
        model = copy.deepcopy(self.model)
        ypred = model.predict(x_test_df)
        main_acc = self.accuracy(y_text_df, ypred)
        print('Non-FeatureBridge model accuracy is: {}\n'.format(main_acc))

        acc_results = []
        test_results = []
        features_to_drop = []

        print('FeatureBridge Pyramid accuracy for reference:')
        for feature in (list(self.feature_map['deviation'].keys()) + list(self.feature_map['level'].keys())):
            features_to_drop.append(feature)
            xtest_x = x_test_df.drop(features_to_drop, axis=1)
            ypred = self.predict(xtest_x)
            test_results.append(ypred)
            acc = self.accuracy(y_text_df, ypred)
            acc_results.append(acc)
            print(f'-- {acc} Accuracy -- when {features_to_drop} was missing')

        x_ax = range(len(test_results[0]))
        for count, result in enumerate(test_results):
            plt.plot(x_ax, result, linewidth=1, label=count)
        plt.title("y-test and y-predicted data distribution")
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend(loc='best', fancybox=True, shadow=True)
        plt.grid(True)
        plt.show()

        acc_results = []
        test_results = []
        features_to_drop = []
        for feature in (list(self.feature_map['deviation'].keys()) + list(self.feature_map['level'].keys())):
            xtest_x = x_test_df.drop(feature, axis=1)
            ypred = self.predict(xtest_x)
            test_results.append(ypred)
            acc = self.accuracy(y_text_df, ypred)
            acc_results.append(acc)

        print("FeatureBridge feature accuracy impact for reference:\nThis shows the impact of each feature when it's missing\n---")
        features = (list(self.feature_map['deviation'].keys()) + list(self.feature_map['level'].keys()))
        plt.bar(features, (acc_results - main_acc))
        plt.xlabel('Feature')
        plt.ylabel('Accuracy')
        plt.title('Features and their accuracy related')
        plt.show()

        print("FeatureBridge performance matrix:\nThis shows the performance of FeatureBridge, the average accuracy for every number of features missing.\n---")
        acc_results = []
        test_results = []
        features_to_drop = []
        all_combinations = []
        list_combinations = []
        main_accuracy = []
        base_f = (list(self.feature_map['deviation'].keys()) + list(self.feature_map['level'].keys()))
        for r in range(1, len(base_f)):
            acc_results = []
            test_results = []
            features_to_drop = []
            all_combinations = []
            list_combinations = []
            all_combinations = combinations(base_f, r)
            for comb in list(all_combinations):
                list_combinations.append(list(comb))
            for feature in list_combinations:
                xtest_x = x_test_df.drop(feature, axis=1)
                ypred = self.predict(xtest_x)
                test_results.append(ypred)
                acc = self.accuracy(y_text_df, ypred)
                acc_results.append(acc)
            avg = sum(acc_results) / len(acc_results)
            main_accuracy.append(avg)
            print('Average FeatureBridge accuracy with {} missing features: {}'.format(r, avg))

        x_bx = range(1, len(main_accuracy) + 1)
        plt.plot(x_bx, main_accuracy, linewidth=1, label='accuracy')
        plt.title("accuracy by number of missing features")
        plt.xlabel('Number of missing features')
        plt.ylabel('Accuracy')
        plt.legend(loc='best', fancybox=True, shadow=True)
        plt.grid(True)
        plt.show()
