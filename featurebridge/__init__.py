#!/bin/env python
# featurebridge/__init__.py
__author__ = "Netanel Eliav"
__email__ = "inetanel@me.com"
__version__ = "1.0.0 alpha"
__license__ = "MIT License"

# module level doc-string
__doc__ = """
featurebridge
=====================================================================
FeatureBridge is a revolutionary adaptive modeling for machine learning applications,
particularly in the realm of Artificial Intelligence. It tackles a common challenge in AI projects:
handling missing features in real-world scenarios. Machine learning models are often trained on specific features,
but when deployed, users may not have access to all those features for predictions.
FeatureBridge bridges this gap by enabling models to intelligently predict and fill in missing features, similar to how humans handle incomplete data.
This ensures that AI models can seamlessly manage missing data and features while providing accurate predictions.

### Key Features

- Missing Feature Prediction:** FeatureBridge empowers AI models to predict and fill in missing features based on the available data.
- Feature Selection for Mapping:** You can impact the features prediction methods by using configurable thresholds for importance, correlation, and accuracy.
- Adaptive Modeling:** Utilize machine learning models to predict missing features, maintaining high prediction accuracy even with incomplete data.
- Custom Accuracy Logic:** Define your own accuracy calculation logic to fine-tune feature selection.
- Feature Distribution Handling:** Automatically determine the best method for handling feature distribution based on data characteristics.
- Dependency Management:** Identify mandatory, deviation, and leveled features to optimize AI model performance.
"""

# Import FeatureBridge classes, methods, and helpers
from .featurebridge import (
    FeatureBridge,
)  # Import the FeatureBridge class from the featurebridge module

__all__ = [  # List of symbols to be exported when using "from featurebridge import *"
    "FeatureBridge",  # Export the FeatureBridge class
]
