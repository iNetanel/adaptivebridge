<p align="center">
  <a href="https://inetanel.github.io/adaptivebridge">
    <img src="https://github.com/inetanel/adaptivebridge/blob/main/docs/assets/images/wide_logo.jpeg" width="600" />
  </a>
</p>

# AdaptiveBridge CHANGELOG
 - **Package Name**: AdaptiveBridge
 - **Author**: Netanel Eliav
 - **Author Email**: netanel.eliav@gmail.com
 - **License**: MIT License

All notable changes to this package will be documented in this file.

The Changelog trailer accepts the following values:
 - **Feature**: New feature.
 - **Enhancement**: Enhancement to feature or core capability.
 - **Bug Fix**: Bug fix.
 - **Change**: Any change.
 - **Deprecation**: New deprecation.
 - **Remove**: Feature or component removal.
 - **Security**: Security fix.
 - **Performance**: Performance Improvement.
 - **Note**: General Information.
 - **Other**: Other


### Unreleased:

> - Multi-model selection. This feature will be released in the next version. (delayed)

### Changelog:

- **1.1.0** [290124] (Netanel Eliav):
  1. **Bug Fix**:
    - Fixed typos and addressed an issue where build.sh wasn't generating requirements.txt when the file exists.
  2. **Performance**:
    - Improved model mapping (part of the .fit method) by restructuring some methods, resulting in fewer iterations for the same tasks.
  3. **Feature**:
    - Added a new method for adaptive mode, the new Mutually Exclusive Estimation (_mutually_exclusive_detection) for boolean and one-hot encoded features. It will now handle classification tasks better by checking mutually exclusive features that are always the same or always not the same. Please refer to the [documentation](https://inetanel.github.io/adaptivebridge/techniques-and-algorithms.html).
    - Added a progress bar using tqdm lib for .fit method. It now shows progress within AdaptiveBridge for the following processes:
      - Determining Feature Distribution
      - Calculating Feature Importance
      - Estimating Mutually Exclusive
      - Mapping Models and Features
      - Creating Feature Dependencies
  > **NOTE**: This version is the first production release version. It contains unit tests, bug fixes, and a complete set of features to handle most ML models.

---
- **1.0.1 - Beta** [190123] (Netanel Eliav):
  1. **Bug Fix**: Fixed benchmark not showing all features.
  2. **Enhancement**: Added colors to graphs; now mandatory features are in blue, and the rest are in green.

---
- **1.0.0 - Beta** [180123] (Netanel Eliav):
  1. **Feature**:
    - Added unit tests and L1 (E2E) tests meeting production-ready standards, reaching 98% code coverage.
    > **NOTE**: This version is the initial pre-production beta release version; please refer to the documentation.
  2. **Enhancement**:
    - Added `_mode_selector` method for discrete and continuous distributions. It now handles multi-mode edge cases better.
    - Added `_DistributionType` class to manage all types of data distributions more effectively.
    - Added comments for better code review.
  3. **Remove**:
    - Removed all unique central tendency for each data distribution. Now it has a new generic algorithm for all types of distribution.
  4. **Bug Fix**:
    - Fixed `__str__` to show Accuracy Logic name and not object name.
    - Fixed `predict` method raising an error in edge cases; now it executes feature engineering mismatch check first before executing the bridge method.
    - Fixed `feature_importance_score` method not executable; now it requires a dataset as a parameter.
  5. **Change**:
    - Changed author email to the updated one.

---
- **0.9.1 - Alpha** [021023] (Netanel Eliav):
  1. **Bug Fix**: Fixed images and documentation; now they show the logos.
  2. **Enhancement**: Improved the PyPi package builder (build.sh); now it takes the version from the CHANGELOG.md dynamically.
  3. **Enhancement**: Added the changelog to the PyPi package page.

---
- **0.9.0 - Alpha** [011023] (Netanel Eliav):
  > **Note**: This version is the initial official alpha version.
  1. **Alpha version**.
    - **Change**: PEP 8 for all files.
    - **Feature**: Support for feature engineering.
    - **Remove**: Pyramid benchmark was removed due to low value and accuracy.
    - **Enhancement**: Added documentation.

---
- **0.8.0 - Alpha** [130923] (Netanel Eliav):
  > **NOTE**: This version is the initial pre-alpha release version; please refer to the documentation.