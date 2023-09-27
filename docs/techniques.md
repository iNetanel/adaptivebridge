# Techniques and Algorithms

## Feature Importance

In the context of FeatureBridge, understanding feature importance is essential for optimizing predictions. Feature importance helps determine which features should be prioritized for prediction, particularly when they meet a certain importance threshold defined by the user (referred to as `importance_threshold`).

Feature importance quantifies the contribution of each feature to a machine learning model's predictions. It's important to note that the importance of a feature is not solely determined by the weight (often denoted as "w") assigned to it in models like linear regression or logistic regression. Instead, it depends on several factors, including magnitude, direction, and impact.

### Magnitude of the Weight (w)

The magnitude of the weight represents the strength of a feature's influence on the model's predictions. Higher absolute weight values indicate a stronger impact. For example, if feature X has a weight of 2, while feature Y has a weight of 0.5, feature X is considered more influential because it has a greater effect.

### Direction of the Weight (w)

The direction of the weight (positive or negative) signifies the influence's direction. A positive weight suggests that increasing the feature's value leads to an increase in the model's prediction, while a negative weight implies the opposite. Understanding the sign of the weight is crucial for interpreting the direction of impact. When assessing pure importance, consider the absolute value of the weight.

### Impact of a Feature

The impact of a feature reflects its actual contribution to predictions. This impact can be estimated using two approaches:

1. **Negative Impact on Model Accuracy:** When a feature is absent, its negative impact on the model's accuracy can be measured. Features with a significant negative impact are crucial.

2. **Portion of Prediction Value:** The contribution of a feature to the prediction can be calculated by multiplying its weight (w) by the feature's value. For instance, if w = 0.5 and x = 100, the feature contributes 50 to the prediction. If the prediction (y) is 500, this feature has a 10% impact on the final prediction.

By using magnitude (w), and impact only, we are calculating the feature's importance and effectiveness.
The feature's importance can be accessed by the instance variable `FeatureBridge.feature_importance_score()`