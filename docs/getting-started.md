# Getting Started

## Installation

You can install FeatureBridge via pip:

```bash
pip install featurebridge

```

Alternatively, you can install FeatureBridge directly from the source code.
Download the package and navigate to the folder:
```shell
   git clone https://github.com/iNetanel/featurebridge.git
   pip install -r requirements.txt

```

## Initialization

### Model Initialization

To begin, make sure to initialize FeatureBridge's dependencies.
You can use many supported models from scikit-learn, such as Linear Regression.
```python
# Import a supported model, for example, Linear Regression.
from sklearn.linear_model import LinearRegression

model = LinearRegression()

```

Currently, FeatureBridge is designed to work with most scikit-learn models.
Refer to the list below for unsupported models:

- `KNeighborsClassifier`

If you wish to use a custom model, ensure it aligns with the scikit-learn interface, implementing basic methods like `.fit`, `.conf_` (or .feature_importances_), and `.predict`.

### Initializing FeatureBridge

To start using FeatureBridge, import the `FeatureBridge` class from the featurebridge library:
```python
# Import the FeatureBridge class from the featurebridge library.
from featurebridge import FeatureBridge

```

Initialize the FeatureBridge class by providing the following parameters:

- `model` (Mandatory): The machine learning model (e.g., LinearRegression) for modeling.
- `correlation_threshold`: The correlation threshold for feature selection based on correlation.
- `min_accuracy`: The minimum accuracy required for feature selection.
- `default_accuracy_selection`: The default accuracy threshold for feature selection.
- `importance_threshold`: The threshold for feature importance.
- `accuracy_logic` (Optional): Custom accuracy calculation logic.

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

# Example: Model Initialization
Here are examples of how to initialize FeatureBridge:
```python
# Example: Basic Initialization
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression

feature_bridge = FeatureBridge(LinearRegression())

```

```python
# Example: Advanced Initialization
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score  # Another method for accuracy that can be used

feature_bridge = FeatureBridge(LinearRegression(), correlation_threshold=0.25, min_accuracy=0.5, default_accuracy_selection=0.95, importance_threshold=0.1, accuracy_logic=r2_score)

```

### Training (Fitting) FeatureBridge

To use FeatureBridge, you need to train (fit) it to a complete dataset without missing features or NaN values,
including both the features and the target (y) separately. Here's how to do it:

See below how to train (fit) FeatureBridge model:
```python
# Example: Fit method
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression

feature_bridge = FeatureBridge(LinearRegression())

# x_df: A data structure like a Pandas DataFrame containing the features.
# y_df: A data structure like a Pandas Series containing the target (y).
feature_bridge.fit(x_df, y_df)

# Now the feature_bridge instance contains all the necessary information, the Adaptive Model, the statistics, and the ability to predict and bridge any gaps in future datasets.

```

### Making Predictions using FeatureBridge

To make predictions with FeatureBridge, you have two options:
1. Using FeatureBridge's built-in `.predict` method, which utilizes the Adaptive Model(s) along with the core Bridge capabilities automatically.
2. ing FeatureBridge's built-in `.bridge` method, which completes any missing features and NaN values in your dataset.

By using the `.predict` method, you'll have a faster and straightforward way to predict any dataset.
Please see [Making Predictions](/featurebridge/methods-and-parameters.html#making-predictions) for more details.

By using the `.bridge` method, you'll have the core value of FeatureBridge, providing you with the data that includes any missing data, allowing for more flexibility in your next steps. Please see [Bridging Missing Data](/featurebridge/methods-and-parameters.html#bridging-missing-data) for details.

Here's how to predict using FeatureBridge:
```python
# Example: Predict using built-in method
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression

feature_bridge = FeatureBridge(LinearRegression())

# x_df: A data structure like a Pandas DataFrame containing the features.
# y_df: A data structure like a Pandas Series containing the target (y).
feature_bridge.fit(x_df, y_df)

# x_df_new: A data structure like a Pandas DataFrame containing the new features to predict.
ypred = feature_bridge.predict(x_df_new)

# Now ypred will contain the predictions. FeatureBridge allows you to predict data even when it has missing features (fully or partially).

```
By using `.bridge`, you will only get the fixed and completed dataset.
Here's how to predict using FeatureBridge only to complete the dataset:

```python
# Example: Fit method
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression

other_model = LinearRegression()
feature_bridge = FeatureBridge(LinearRegression())

# x_df: A data structure like a Pandas DataFrame containing the features.
# y_df: A data structure like a Pandas Series containing the target (y).
feature_bridge.fit(x_df, y_df)

# x_df_new: A data structure like a Pandas DataFrame containing the new features to predict.
x_df_complete = feature_bridge.bridge(x_df_new)

# To predict using another model or methods.
ypred = other_model.predict(x_df_complete)

# Now x_df_complete will contain the x_df_new dataset but completed, allowing it to be predicted using other models and not just FeatureBridge.

```

### Predicting with FeatureBridge along with Feature Engineering

When using FeatureBridge, you aim to address data issues related to external data sources. Feature Engineering, of any kind, is based on a complete dataset containing the base features.

During the Feature Engineering phase, which is common in most ML projects, you'll need to input the data engineering feature names into FeatureBridge, allowing the model to classify those features as engineered ones.

Here's how to predict using FeatureBridge and enable Feature Engineering:
```python
# Example: Predict using `.bridge` method only (in the future, FeatureBridge will also support user-input feature engineering functions, and the `.predict` method will be used for this)
import featurebridge as featurebridge
from sklearn.linear_model import LinearRegression

other_model = LinearRegression()
feature_bridge = FeatureBridge(LinearRegression())

# x_df: A data structure like a Pandas DataFrame containing the features.
x_df['feature1-Polynomial'] = x_df['feature1']**2  # An example of a feature generated using feature engineering
x_df['feature2-Polynomial'] = x_df['feature2']**2  # An example of a feature generated using feature engineering

# x_df: Now it will be a data structure like a Pandas DataFrame containing the features, including the features from the feature engineering phase.
# y_df: A data structure like a Pandas Series containing the target (y).

# .fit(*, x_df, y_df, feature_engineering=[FEATURE_ENGINEERING_LIST])
feature_engineering_list = ['feature1-Polynomial', 'feature2-Polynomial']
feature_bridge.fit(x_df, y_df, feature_engineering=feature_engineering_list)

# x_df_new: A data structure like a Pandas DataFrame containing the new dataset to be predicted. It will include only the features and not the target (y).
x_df_complete = feature_bridge.bridge(x_df_new)

x_df_complete['feature1-Polynomial'] = x_df_complete['feature1']**2
x_df_complete['feature2-Polynomial'] = x_df_complete['feature2']**2

# Now x_df_complete will contain the x_df_new dataset but completed, including the feature engineering, which can be predicted using other models and not just FeatureBridge.

ypred = feature_bridge.predict(x_df_complete)

# Now x_df_complete will contain the x_df_new dataset but completed, allowing it to be predicted using FeatureBridge. This model uses the features from feature engineering as well.

```

Please note that when using `.bridge`, the Feature Engineering stage needs to come after the dataset is completed. Providing these features beforehand in the new dataset to be predicted will not work because this dataset may have gaps, missing features, or NaN values for some data cells.

### General Example of How to Use FeatureBridge

```python
import featurebridge as featurebridge
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('cell_samples.csv', header=0, sep=',')
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

# Check the benchmark and see the features that can be covered by FeatureBridge
feature_bridge.feature_sequence()
feature_bridge.benchmark(xtrain, ytrain)

```

```python
'''
This stage is to be used on a separate dataset that needs to be predicted.
'''

dfnew = pd.read_csv('data/cell_samples-missing-features.csv', header=0, sep=',')  # New dataset with missing features
y_name = 'Class'
dfnew = dfnew.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
dfnew = dfnew.dropna()
y_df = dfnew[y_name]
dfnew = dfnew.drop([y_name], axis=1)

dfnew = feature_bridge.bridge(dfnew)  # Complete the dataset

# Feature engineering
dfnew['MargAdh2'] = dfnew['MargAdh']**2
dfnew['UnifShape2'] = dfnew['UnifShape']**2
dfnew['BareNuc2'] = dfnew['BareNuc']**2
dfnew['BlandChrom2'] = dfnew['BlandChrom']**2
dfnew['SingEpiSize2'] = dfnew['SingEpiSize']**4

# Predict
ypred = feature_bridge.predict(dfnew)
print(ypred)

```