# Initialization

## Model Initialization

To get started, you will need to make sure you initialize FeatureBridge's dependencies.
```python
# use sklearn supported models, for example, the linear regression model.
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```
At this moment, FeatureBridge intends to work with sklearn linear models (`sklearn.linear_model`) and it supports the next algorithms:

- `SGDRegressor` Please note that using FeatureBridge with SGDR will create a significant performance impact in a matter of training time.
- `LinearRegression`
- `Ridge`
- `Lasso`
- `LassoCV`
- `ElasticNet`
- `BayesianRidge`

_If a custom model needs to be in use, please make sure it will be a linear regression type of model and the sklearn interface will be implemented (exp: `.fit`, `._conf`, `.predict`)

## Initialize FeatureBridge

To get started with FeatureBridge, you need to import the FeatureBridge class from featurebridge lib:
```python
# Import FeatureBridge class from featurebridge lib.
from featurebridge import FeatureBridge
```

Then you need to initialize the `FeatureBridge` class by providing the following parameters:

- `model`: The machine learning model (e.g., LinearRegression) to be used for modelling.
- `correlation_threshold`: The correlation threshold for feature selection based on correlation.
- `min_accuracy`: The minimum accuracy required for feature selection.
- `default_accuracy_selection`: The default accuracy threshold for feature selection.
- `importance_threshold`: The threshold for feature importance.
- `accuracy_logic`: Custom accuracy calculation logic (optional).

```
feature_bridge = FeatureBridge(
    model=model,
    correlation_threshold=0.3,
    min_accuracy=0.5,
    default_accuracy_selection=0.95,
    importance_threshold=0.1,
    accuracy_logic=None
)
```