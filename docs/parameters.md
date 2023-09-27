# Parameters

#### `model`:
The model parameter is according to what was previously mentioned: [Initialization](#initialization)

#### `correlation_threshold`:
The correlation_threshold parameter needs to be set by the user (`default=0.25`).
The parameter set to FeatureBridge is the threshold for the decision in model selection. It means that a higher threshold means more features will not be used Data Distribution method when the Adaptive model is not able to predict, this will make more features mandatory.

A lower threshold will impact the fit performance but it will lower the cases when features are mandatory.

#### `min_accuracy`:
The min_accuracy is a parameter that needs to be set by the user (`default=0.5`).
It sets the minimum accuracy of adaptive model prediction to predict a feature (if missing), if a model cannot predict a feature with higher accuracy than the min_accuracy, it will be set as mandatory.

This means that higher `min_accuracy` will lead to more features that are mandatory and cannot be missing.
Please note: min_accuracy is not the overall accuracy by the FeatureBridge, it's a decision parameter only. please use the `benchmark` method for general accuracy.

#### `default_accuracy_selection`:
The default_accuracy_selection is a parameter that needs to be set by the user (`default=0.95`).
It sets the break point for satisfactory accuracy of adaptive model prediction to predict a feature (if missing), if a model is reaching default_accuracy_selection or higher in accuracy it will stop and break the fitting for that feature.

A lower `default_accuracy_selection` will lead to lower training time.

#### `importance_threshold`:
The importance_threshold is a parameter that needs to be set by the user (`default=0.1`).
The parameter sets the threshold to FeatureBridge for the feature sequence dependencies. It means that a higher `importance_threshold` will lead to more mandatory features. It will tell FeatureBridge that this feature is important enough and that using the Data Distribution method is not satisfactory.

#### `accuracy_logic`:
The accuracy_logic is a parameter that it's optional to be set by the user (`default=None`).
FeatureBridge allows for customization through the custom accuracy calculation logic (`accuracy_logic`), the accuracy logic method will affect many calculations and algorithms within FeatureBridge, from feature selection, to the accuracy of the Adaptive model by using the `benchmark` method.

You can define a custom function to calculate prediction accuracy.
Examples are: `mean_squared_error`, `r2_score`, and `mean_absolute_percentage_error` by sklearn.
Please note, `mean_absolute_percentage_error` and `mean_squared_error` need adjustments to be (1-mean_absolute_percentage_error), please use the custom to tune the sklearn method. We recommend to use `r2_score` or `None`.

You can have any custom method.

See below how to implement custom logic:
```python
# Example 1: Feature selection and model fitting
feature_bridge = FeatureBridge(model=LinearRegression())
feature_bridge.fit(x_train, y_train)

# Example 2: Making predictions with the fitted model
predictions = feature_bridge.predict(x_test)

# Example 3: Custom accuracy logic
def custom_accuracy(y_true, y_pred):
    # Define your custom accuracy calculation here
    pass

feature_bridge = FeatureBridge(model=LinearRegression(), accuracy_logic=custom_accuracy)
feature_bridge.fit(x_train, y_train)
predictions = feature_bridge.predict(x_test)
```