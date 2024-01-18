#!/bin/env python
# adaptivebridge/utils/_error_handler.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Define a custom exception class


class MandatoryFeatureError(Exception):
    """
    Basic Error Handler class for missing mandatory features
    """

# Define a custom exception class


class EngineeringFeatureError(Exception):
    """
    Basic Error Handler class for missing feature-engineering features
    """
