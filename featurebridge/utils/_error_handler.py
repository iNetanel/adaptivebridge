#!/bin/env python
# featurebridge/utils/_error_handler.py
"""
    Package Name: FeatureBridge
    Author: Netanel Eliav
    Author Email: inetanel@me.com
    License: MIT License
    Version: Please refer to the repository for the latest version and updates.
"""


# Define a custom exception class
class MandatoryFeatureError(Exception):
    pass


# Define a custom exception class
class EngineeringFeatureError(Exception):
    pass
