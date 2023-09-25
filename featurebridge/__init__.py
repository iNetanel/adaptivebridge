#!/bin/env python
# featurebridge/__init__.py

"""
    Package Name: FeatureBridge
    Author: Netanel Eliav
    Author Email: inetanel@me.com
    License: MIT License
    Version: Please refer to the repository for the latest version and updates.
"""

# Import FeatureBridge classes, methods, and helpers
from .featurebridge import (
    FeatureBridge,
)  # Import the FeatureBridge class from the featurebridge module

__all__ = [  # List of symbols to be exported when using "from featurebridge import *"
    "FeatureBridge",  # Export the FeatureBridge class
]

__version__ = "1.0.0 alpha"  # The actual version number
