# Methods

## Fitting FeatureBridge Adaptive Model

After initialization, you can fit the adaptive model to your data using the fit method. Provide the feature data frame (`x_df`) and the target variable data (`y_df`) as parameters.
```python
# Fit the model
feature_bridge.fit(x_df, y_df)
```

## Making Predictions

Once the model is fitted, you can make predictions using the `.predict` method. Provide the feature data frame for prediction (`x_df`) as a parameter.
```python
# Make predictions
predictions = feature_bridge.predict(x_df)
```

## Complete and Bridge a DataFrame

Although it's part of the prediction process method, you can choose only to predict missing values and features in a dataset and to have a completely new data frame (dataset) without missing values by using the `.df_bridge` method. Provide the feature data frame to complete (`x_df`) as a parameter.
```python
# Complete data frame (dataset)
complete_x_df = feature_bridge.df_bridge(x_df)
```

## Feature Importance
Feature importance can be assessed using the `feature_importance_score` method, which summarizes the feature importance scores.
```python
feature_bridge.feature_importance_score()
```

Output Example:
```
Feature: crim (0), Score: 0.39030
Feature: zn (1), Score: 0.52751
Feature: indus (2), Score: 0.22896
Feature: chas (3), Score: 0.18584
Feature: nox (4), Score: 9.85505
Feature: rm (5), Score: 23.94361
Feature: age (6), Score: 0.04747
Feature: dis (7), Score: 5.59984
Feature: rad (8), Score: 2.92259
Feature: tax (9), Score: 5.03544
Feature: ptratio (10), Score: 17.58346
Feature: b (11), Score: 3.32124
Feature: lstat (12), Score: 6.63980
```

The above example shows the feature's importance **not in** percentage but by absolute value,
in our example, Feature rm (5) is providing an absolute value of 23.94 to the prediction target.

## Feature Sequence Dependencies
Feature Sequence Selection is a crucial aspect of FeatureBridge. It automatically identifies mandatory features, and methods to handle deviations in data distribution, and creates predictive models for features. The process involves the following steps:

- Identifying mandatory and deviation features.
- Performing feature sequence based on dependencies.

Feature Sequence Dependencies can be displayed by the `feature_sequence` method that will print the features sequence, dependencies, and their handling methods.

```python
feature_bridge.feature_sequence()
```

Output Example:
```
Feature Sequence Dependencies:
Mandatory: (Must be provided by the user)
 - Feature crim
 - Feature zn

Data Distribution Method: (data distribution method will be used and not prediction)
 - Feature chas, 0

Prediction by Adaptive Model: (will be predicted by adaptive model)
 - Feature rm, Dependencies: ['zn']
 - Feature ptratio, Dependencies: ['crim', 'zn', 'rm']
 - Feature nox, Dependencies: ['crim', 'zn', 'rm']
 - Feature b, Dependencies: ['crim', 'nox']
 - Feature tax, Dependencies: ['crim', 'zn', 'nox', 'ptratio', 'b']
 - Feature age, Dependencies: ['crim', 'zn', 'nox', 'tax', 'b']
 - Feature dis, Dependencies: ['crim', 'zn', 'nox', 'age', 'tax', 'b']
 - Feature indus, Dependencies: ['crim', 'zn', 'nox', 'rm', 'dis', 'tax', 'b']
 - Feature lstat, Dependencies: ['crim', 'zn', 'industry', 'rm', 'age', 'tax', 'b']
 - Feature rad, Dependencies: ['crim', 'zn', 'industry', 'nox', 'age', 'tax', 'ptratio', 'b']
```

The above example shows the features `crim` and `zn` are mandatory, this means they will have to be provided to be able to predict the target.
The example also shows that the feature `chas ` will be completed and predicted using one of the data distribution methods, this means the feature will be given a value (when missing) based on the data distribution, in the example case, it getting `0` because this feature is bool and the mean is below 0.5.
The above example shows the rest of the features and their dependency structure and sequence, in general, it shows how each feature is dependent on other features. the list is short by the sequence. you can see that feature `rm` will be used for feature `zn` for the adaptive model and to be predicted. Feature `ptratio` will use more features, including `rm`, that is why it will be the second feature that will be predicted in case both features `rm` and `ptratio' are missing.

## Benchmark
Benchmark is a very important step for FeatureBridge.the method will allow you to check every aspect of FeatureBridge's structure, performance, and accuracy.

```python
feature_bridge.benchmark(x_test_df, y_text_df)
```

Here is some key information that the benchmark will provide:

**Non-FeatureBridge Model Accuracy:**
This shows the non-FeatureBridge model accuracy.
```bash
Non-FeatureBridge Model Accuracy: 0.8808579769328666`
```

**FeatureBridge Pyramid Accuracy:**
This shows the accuracy when features are missing, one by one and accordingly to the dependency.
```bash
FeatureBridge Pyramid Accuracy:
-- 0.8522360668910814 Accuracy -- when ['crim'] was missing
-- 0.840728799573311 Accuracy -- when ['crim', 'zn'] was missing
-- 0.8438389851603283 Accuracy -- when ['crim', 'zn', 'chas'] was missing
-- 0.7984019354988106 Accuracy -- when ['crim', 'zn', 'chas', 'rm'] was missing
-- 0.7663155541581351 Accuracy -- when ['crim', 'zn', 'chas', 'rm', 'ptratio'] was missing
-- 0.7122044681450528 Accuracy -- when ['crim', 'zn', 'chas', 'rm', 'ptratio', 'nox'] was missing
-- 0.699574698659302 Accuracy -- when ['crim', 'zn', 'chas', 'rm', 'ptratio', 'nox', 'b'] was missing
-- 0.6453239701475657 Accuracy -- when ['crim', 'zn', 'chas', 'rm', 'ptratio', 'nox', 'b', 'tax'] was missing
-- 0.6459538116331353 Accuracy -- when ['crim', 'zn', 'chas', 'rm', 'ptratio', 'nox', 'b', 'tax', 'age'] was missing
-- 0.7176847838906694 Accuracy -- when ['crim', 'zn', 'chas', 'rm', 'ptratio', 'nox', 'b', 'tax', 'age', 'dis'] was missing
-- 0.7208448622936061 Accuracy -- when ['crim', 'zn', 'chas', 'rm', 'ptratio', 'nox', 'b', 'tax', 'age', 'dis', 'indus'] was missing
-- 0.6137469459180667 Accuracy -- when ['crim', 'zn', 'chas', 'rm', 'ptratio', 'nox', 'b', 'tax', 'age', 'dis', 'indus', 'lstat'] was missing
-- 0.665067094620444 Accuracy -- when ['crim', 'zn', 'chas', 'rm', 'ptratio', 'nox', 'b', 'tax', 'age', 'dis', 'indus', 'lstat', 'rad'] was missing
```

****- A Data Distribution Plot for FeatureBridge Pyramid:****
This shows the data plot when features are missing, by the Pyramid sequence.

![1a3d16c0-b901-4ef0-859f-e332a1e82828](https://github.com/iNetanel/featurebridge/assets/69385881/10b716c9-9f4f-46ee-9998-2a7a3bd917fc)

FeatureBridge Features Accuracy Impact:
This shows the impact of each feature when it's missing.

![3cca6118-f9be-4312-9def-b248b0f9939d](https://github.com/iNetanel/featurebridge/assets/69385881/b568a867-adcf-488d-bedf-4f16dccdc2b4)

**FeatureBridge Performance Matrix:**
This shows the performance of FeatureBridge, the average accuracy for every number of features missing. This will include a plot that shows how the accuracy is dropping.

```bash
FeatureBridge performance matrix:
This shows the performance of FeatureBridge, the average accuracy for every number of features missing.
---
Average FeatureBridge accuracy with 1 missing features: 0.845135522250052
Average FeatureBridge accuracy with 2 missing features: 0.8371119809276981
Average FeatureBridge accuracy with 3 missing features: 0.8284419704508165
Average FeatureBridge accuracy with 4 missing features: 0.8189357820863832
Average FeatureBridge accuracy with 5 missing features: 0.8084686792445318
Average FeatureBridge accuracy with 6 missing features: 0.7969622137206069
Average FeatureBridge accuracy with 7 missing features: 0.7843961511171201
Average FeatureBridge accuracy with 8 missing features: 0.7708000830336739
Average FeatureBridge accuracy with 9 missing features: 0.7562430900808489
Average FeatureBridge accuracy with 10 missing features: 0.7408298248251479
Average FeatureBridge accuracy with 11 missing features: 0.7246676865310517
Average FeatureBridge accuracy with 12 missing features: 0.7077357069383757
```
![9ba30d77-8557-4cb0-91fe-7ad0a95be5ce](https://github.com/iNetanel/featurebridge/assets/69385881/ebcab62f-6de7-467b-8ae5-921312bca9b0)

The performance of FeatureBridge should be according to the **FeatureBridge performance Matrix** ONLY.