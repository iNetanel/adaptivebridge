#!/bin/env python
# adaptivebridge/__init__.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# module level doc-string
__doc__ = """
adaptivebridge
=====================================================================
AdaptiveBridge is a revolutionary adaptive modeling for machine learning applications,
particularly in the realm of Artificial Intelligence. It tackles a common challenge in AI projects:
handling missing features in real-world scenarios. Machine learning models are often trained on specific features,
but when deployed, users may not have access to all those features for predictions.
AdaptiveBridge bridges this gap by enabling models to intelligently predict and fill in missing features, similar to how humans handle incomplete data.
This ensures that AI models can seamlessly manage missing data and features while providing accurate predictions.

### Key Features

- Missing Feature Prediction:** AdaptiveBridge empowers AI models to predict and fill in missing features based on the available data.
- Feature Selection for Mapping:** You can impact the features prediction methods by using configurable thresholds for importance, correlation, and accuracy.
- Adaptive Modeling:** Utilize machine learning models to predict missing features, maintaining high prediction accuracy even with incomplete data.
- Custom Accuracy Logic:** Define your own accuracy calculation logic to fine-tune feature selection.
- Feature Distribution Handling:** Automatically determine the best method for handling feature distribution based on data characteristics.
- Dependency Management:** Identify mandatory, deviation, and leveled features to optimize AI model performance.
"""

# Import AdaptiveBridge classes, methods, and helpers
from .adaptivebridge import (
    AdaptiveBridge,
)  # Import the AdaptiveBridge class from the adaptivebridge module

__all__ = [  # List of symbols to be exported when using "from adaptivebridge import *"
    "AdaptiveBridge",  # Export the AdaptiveBridge class
]
