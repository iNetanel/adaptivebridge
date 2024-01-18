"""
Package Name: AdaptiveBridge
Author: Netanel Eliav
Author Email: netanel.eliav@gmail.com
License: MIT License
Version: Please refer to the repository for the latest version and updates.
"""

from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as req_file:
    requirements = req_file.read().splitlines()

with open("README.md", encoding="utf-8") as read_file:
    readme = read_file.read()

with open("CHANGELOG.md", "r+") as changelog:
    adaptivebridge_version = changelog.read().split(
        "---")[0].split("- **")[-1][:5]


setup(
    name="adaptivebridge",
    version=adaptivebridge_version,
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.2", "twine>4.0.2"],
    },
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.7",
    author="Netanel Eliav",
    author_email="netanel.eliav@gmail.com",
    url="https://github.com/inetanel/adaptivebridge",
    author_url="https://inetanel.com",
    description="Revolutionizing ML adaptive modelling for handling missing features and data. The model can predict missing data in real-world scenarios.",
    long_description=readme,
    long_description_content_type="text/markdown",  # Specify Markdown format
    keywords=[
        "sklearn",
        "scikit-learn",
        "python",
        "data analysis",
        "machine learning",
        "data visualization",
        "python library",
        "data processing",
        "data science",
        "data exploration",
        "data manipulation",
        "analytics",
        "statistics",
        "artificial intelligence",
        "AI",
        "feature engineering",
        "data preprocessing",
        "predictive modeling",
        "classification",
        "regression",
        "missing data",
        "data cleaning",
        "data imputation",
        "data quality",
        "missing data analysis",
        "data handling",
        "data integrity",
        "data cleansing",
        "data wrangling",
        "data validation",
        "data completeness",
        "impute missing values",
        "data missingness",
        "missing data detection",
        "data quality assessment",
        "data pre-processing tool",
    ],
    license="MIT",
    project_urls={
        "Author Website": "https://inetanel.com",
        "Documentation": "https://inetanel.github.io/adaptivebridge",
        "Changelog": "https://github.com/iNetanel/adaptivebridge/blob/main/CHANGELOG.md",
        "Source Code": "https://github.com/inetanel/adaptivebridge",
        "Issue Tracker": "https://github.com/inetanel/adaptivebridge/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
