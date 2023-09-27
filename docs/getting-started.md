# Getting Started

## Installation

You can install FeatureBridge via pip:

```bash
pip install featurebridge
```

Or, as another opetion, you can install FeatureBridge directly from the source code.
You will need to use the folder that has been downloaded as a packadge.
```shell
   git clone https://github.com/iNetanel/featurebridge.git
   pip install -r requirements.txt
```

## Initialization

### Model Initialization

To get started, you will need to make sure you initialize FeatureBridge's dependencies.
```python
# use sklearn many supported models, for example, the linear regression Model and Classifier.
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```
At this moment, FeatureBridge intends to work with must of sklearn models,.
Please see below models list that are not supported yet:

- `KNeighborsClassifier`

_If you need to use custom model, please make sure it will aligned with sklearn interface, meaning it will be implemented the basic methods like `.fit`, `.conf_` (or `.feature_importances_`), and `.predict`.

### Initialize FeatureBridge

To get started with FeatureBridge, you need to import the FeatureBridge class from featurebridge lib:
```python
# Import FeatureBridge class from featurebridge lib.
from featurebridge import FeatureBridge
```

Then you need to initialize the `FeatureBridge` class by providing the following parameters:

- `model` (Mandatory): The machine learning model (e.g., LinearRegression) to be used for modelling.
- `correlation_threshold`: The correlation threshold for feature selection based on correlation.
- `min_accuracy`: The minimum accuracy required for feature selection.
- `default_accuracy_selection`: The default accuracy threshold for feature selection.
- `importance_threshold`: The threshold for feature importance.
- `accuracy_logic` (Optional): Custom accuracy calculation logic (optional).

```
feature_bridge = FeatureBridge(
    model=model,
    correlation_threshold=0.25,
    min_accuracy=0.5,
    default_accuracy_selection=0.95,
    importance_threshold=0.1,
    accuracy_logic=None
)
```

# Example : Model Initialization
See below how to Initialized FeatureBridge:
```python
# Example : Basic Initialization
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression
feature_bridge = FeatureBridge(LinearRegression())
```

```python
# Example : Advanced initialization
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score # another method for accuracy that can be use.
feature_bridge = FeatureBridge(LinearRegression(), correlation_threshold=0.25, min_accuracy=0.5, default_accuracy_selection=0.95, importance_threshold=0.1, accuracy_logic=r2_score)
```

### Train (Fit) FeatureBridge

You use FeatureBridge you need to train (to fit) FeatureBridge to a complete dataset without missing features or NaN(s), that includes the features and the target (y) seperatly:

See below how to train (fit) FeatureBridge model:
```python
# Example : Fit method
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression
feature_bridge = FeatureBridge(LinearRegression())
# x_df == A data structure like Pandas DataFrame, that will contain the features.
# y_df == A data structure like Pandas Series, that will contain the target (y).
feature_bridge.fit(x_df, y_df)

# Now feature_bridge instance contain all the information, the Adaptive Model, the statistics and the ability to predict and to 'bridge' any gap in future dataset
```

### Predict using FeatureBridge

In order to use FeatureBridge Prediction, you have two ways:
1. Using FeatureBridge tobuilt-in `.predict` method that will use the Adaptive Model(s) along with the Bridge core capabilities automatically.
2. Using FeatureBridge built-in `.bridge` method that will complete any missing features and NaN(s) in your dataset.

By using the `.predict` method you will have faster and stright forward way to predcit any dataset.
Please see [Making Predictions](/featurebridge/methods-and-parameters.html#making-predictions)

By using the `.bridge` method, you will have the core value of FeatureBridge providing you the data with any missing data to move forward to your next step in more flexible way.
Please see [Bridging Missing Data](/featurebridge/methods-and-parameters.html#bridging-missing-data).

See below how to predict using FeatureBridge:
```python
# Example : Predict using built-in method.
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression
feature_bridge = FeatureBridge(LinearRegression())
# x_df == A data structure like Pandas DataFrame, that will contain the features.
# y_df == A data structure like Pandas Series, that will contain the target (y).
feature_bridge.fit(x_df, y_df)
# x_df_new == A data structure like Pandas DataFrame, that will contain the features.
ypred = feature_bridge.predict(x_df_new)

# Now ypred will contain the predictions, by using FeatureBridge you will be able to predict data even when it will missing features (fully or partially)
```

By using `.bridge` you will only get the fixed and completed dataset.
See below how to predict using FeatureBridge only to complete dataset:
```python
# Example : Fit method
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression
other_model = LinearRegression()
feature_bridge = FeatureBridge(LinearRegression())
# x_df == A data structure like Pandas DataFrame, that will contain the features.
# y_df == A data structure like Pandas Series, that will contain the target (y).
feature_bridge.fit(x_df, y_df)
# x_df_new == A data structure like Pandas DataFrame, that will contain the features.
x_df_complete = feature_bridge.bridge(x_df_new)

# To predict using other model or methods.
ypred = other_model.predict(x_df_complete)

# Now x_df_complete will contain the x_df_new dataset but complete one that can be predicted using other models and not featureBridge.
```

### Predict using FeatureBridge with Feature Engineering

When using FeatureBridge, you are aiming to solve data issues that related to extranal data sources.
Feature Engineering (any kind) is based on a complete dataset that contain the base features.

When having the Feature Engineering phase (usually with every ML project), you will need to input the data engineering features names to FeatureBridge to let the model classify those features as engineering ones.

See below how to predict using FeatureBridge and allow Feature Engineeting:
```python
# Example : Predict using `.bridge` method only (in the future FeatureBridge will support also the feature engineering funnction to be inputed by the user. in that way the `.predict` method will be in used)
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression
other_model = LinearRegression()
feature_bridge = FeatureBridge(LinearRegression())
# x_df == A data structure like Pandas DataFrame, that will contain the features.
x_df['feature1-Polinomal'] = x_df['feature1']**2 # en example of feature that was generated using feature engineering
x_df['feature2-Polinomal'] = x_df['feature2']**2 # en example of feature that was generated using feature engineering
# x_df == Now it will be data structure like Pandas DataFrame, that will contain the features including the features from feature engineeting phase .
# y_df == A data structure like Pandas Series, that will contain the target (y).
# .fit(*, x_df, y_df, feature_engineering=[FEATURE_ENGINEERING_LIST])
feature_engineering_list = ['feature1-Polinomal', 'feature2-Polinomal']
feature_bridge.fit(x_df, y_df, feature_engineering=feature_engineering_list)
# x_df_new == A data structure like Pandas DataFrame, that will contain the new dataset to be predicted. it will included only the features and not the target (y).
x_df_complete = feature_bridge.bridge(x_df_new)

x_df_complete['feature1-Polinomal'] = x_df_complete['feature1']**2
x_df_complete['feature2-Polinomal'] = x_df_complete['feature2']**2

# Now x_df_complete will contain the x_df_new dataset but complete including the feature engineering that can be predicted using other models and not featureBridge.

ypred = feature_bridge.predict(x_df_complete)

# Now x_df_complete will contain the x_df_new dataset but complete one that can be predicted using FeatureBridge. this model using the features from feature engineeting also.
```

Please note: when using the `.bridge` the Feature Engineering stage need also to come after the dataset is completed, by providing this before in the new dataset that need to be predicted, it will not work because this dataset may have gaps, missing features, or NaN(s) for some data cells.

### General Example for how to use FeatureBridge

```python
import featurebridge as featurebridge
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('cell_samples.csv', header = 0,  sep=',')
y_name = 'Class'
df = df.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
df = df.dropna()
x_df = df.drop([y_name], axis=1)
y_df = df[y_name]

# Basic feature engineering
x_df['MargAdh2'] = x_df['MargAdh']**2
x_df['UnifShape2'] = x_df['UnifShape']**2
x_df['BareNuc2'] = x_df['BareNuc']**2
x_df['BlandChrom2'] = x_df['BlandChrom']**2
x_df['SingEpiSize2'] = x_df['SingEpiSize']**2

xtrain, xtest, ytrain, ytest = train_test_split(x_df, y_df, test_size=0.15)

# Activate FeatureBridge model based on DecisionTreeClassifier
feature_bridge = featurebridge.FeatureBridge(model, correlation_threshold=0.25, min_accuracy=0.5, default_accuracy_selection=0.95, importance_threshold=0.1, accuracy_logic=None)
feature_bridge.fit(xtrain, ytrain, feature_engineering=['MargAdh2', 'UnifShape2', 'BareNuc2', 'BlandChrom2', 'SingEpiSize2'])

xtrain, xtest, ytrain, ytest = train_test_split(x_df, y_df, test_size=0.15)

# Fit the model
feature_bridge.fit(xtrain, ytrain)

# Check Benchmark and see the features that can be covered by FeatureBridge
feature_bridge.feature_sequence()
feature_bridge.benchmark(xtrain, ytrain)
```

```python
'''
This stage is to use on seperate dataset that need to be predict.
'''

dfnew = pd.read_csv('data/cell_samples-missing-features.csv', header = 0,  sep=',') # New dataset with missing features
y_name = 'Class'
dfnew = dfnew.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
dfnew = dfnew.dropna()
y_df = dfnew[y_name]
dfnew = dfnew.drop([y_name], axis=1)

dfnew = feature_bridge.bridge(dfnew) # complete the dataset

# feature engineering
dfnew['MargAdh2'] = dfnew['MargAdh']**2
dfnew['UnifShape2'] = dfnew['UnifShape']**2
dfnew['BareNuc2'] = dfnew['BareNuc']**2
dfnew['BlandChrom2'] = dfnew['BlandChrom']**2
dfnew['SingEpiSize2'] = dfnew['SingEpiSize']**4

# Predict.
ypred = feature_bridge.predict(dfnew)
print(ypred)

```