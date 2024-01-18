#!/bin/env python
# tests/test_init.py

"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

# Import necessary libraries
import pytest
import os


def test_init():
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory path of the current file
    directory_path = os.path.dirname(current_file_path)

    # Join the directory path with the target directory name
    target_directory = os.path.join(directory_path, 'tests_0_utils')

    # Check if the target directory exists
    assert os.path.isdir(
        target_directory), f"Directory does not exist: {target_directory}"


if __name__ == "__main__":
    # Run the tests using pytest if the script is executed directly
    pytest.main()
