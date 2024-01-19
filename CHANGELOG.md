<p align="center">
  <a href="https://inetanel.github.io/adaptivebridge">
  <img src="https://github.com/inetanel/adaptivebridge/blob/main/docs/assets/images/wide_logo.jpeg" width="600" />
  </a>
</p>

# AdaptiveBridge CHANGELOG
 - Package Name: AdaptiveBridge
 - Author: Netanel Eliav
 - Author Email: netanel.eliav@gmail.com
 - License: MIT License

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

> - Multi-model selection. This feature will be released in the next version.

### Changelog:

- **1.0.1 - Beta** [190123] (Netanel Eliav):
  1. Bug Fix: Fixed benchmark not showing all features.
  2. Enhancement: Added colors to graphs; now mandatory features are in blue, and the rest are in green.

---
- **1.0.0 - Beta** [180123] (Netanel Eliav):
  1. Feature:
    - Added unit tests and L1 (E2E) tests meeting production-ready standards, reaching 98% code coverage.
    > Note: This version is the initial pre-production beta release version; please make sure to check the documentation.
  2. Enhancement:
    - Added `_mode_selector` method for district and continuous distributions. Now it will handle multi-mode edge cases better.
    - Added `_DistributionType` class to manage all types of data distributions more effectively.
    - Added comments for everything for better code review.
  3. Remove:
    - Removed all unique central tendency for each data distribution. Now it will have a new generic algorithm for all types of distribution.
  4. Bug Fix:
    - Fixed `__str__` to show Accuracy Logic name and not object name.
    - Fixed `predict` method raising an error in edge cases; now it will execute feature engineering mismatch check first before executing the bridge method.
    - Fixed `feature_importance_score` method not executable; now it requires a dataset as a parameter.
  5. Change:
    - Changed author email to be the updated one.

---
- **0.9.1 - Alpha** [021023] (Netanel Eliav):
  1. Bug Fix: Fixed images and documentation; now it will show the logos.
  2. Enhancement: Improved the PyPi package builder (build.sh); now it will take the version from the CHANGELOG.md dynamically.
  3. Enhancement: Added the changelog to the PyPi package page.

---
- **0.9.0 - Alpha** [011023] (Netanel Eliav):
  > Note: This version is the initial official alpha version.
  1. Alpha version.
    - Change: PEP 8 for all files.
    - Feature: Support for feature engineering.
    - Remove: Pyramid benchmark was removed due to low value and accuracy.
    - Enhancement: Added documentation.

---
- **0.8.0 - Alpha** [130923] (Netanel Eliav):
  > Note: This version is the initial pre-alpha release version; please make sure to check the documentation.
