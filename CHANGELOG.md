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

> - Multi model selection. this feature will be release in the next version.

### Changelog:

- **1.0.0 - Beta** [180123] (Netanel Eliav):
  1. Feature:
    - Add uni-tests and L1 (E2E) tests meeting production ready, reaching 98% code-coverage.
    > Note: this version is initial pre-production beta release version, please make sure to check the documentation.
  2. Enhancement:
    - Add _mode_selector method for district and continuous distributions. Now it will handled better multi-mode edge cases.
    - Add _DistributionType class to manage better all type of data distributions.
    - Add comments for everything for better code review.
  3. Remove:
    - Remove all unique central tendency for each data distribution. now it will have a new generic algorithm for all type of distribution.
  4. Bug Fix:
    - Fix __str__ to show Accuracy Logic name and not object name.
    - Fix predict method raise an error in edge cases, now it will execute feature engineering mis-match check first before executing bridge method.
    - Fix feature_importance_score method not executable, now it will required dataset as parameter.
  5. Change:
   - Change Author email to be updated one.
---
- **0.9.1 - Alpha** [021023] (Netanel Eliav):
  1. Bug Fix - Fix Images and Documentation. now it will show the logos.
  2. Enhancement - Improvement to the PyPi package builder (build.sh). now it will take the version fro the CHANGELOG.md dynamicly.
  3. Enhancement - Added the changelog to the PyPi package page.
---
- **0.9.0 - Alpha** [011023] (Netanel Eliav):
  > Note: this version is initial official alpha version.
  1. Alpha version.
    - Change - PEP 8 for all files.
    - Feature - support for feature engineering.
    - Remove - Pyramid benchmark was remove due to low value and accuracy.
    - Enhancement - Add documentation.
---
- **0.8.0 - Alpha** [130923] (Netanel Eliav):
  > Note: this version is initial pre-alpha release version, please make sure to check the documentation.