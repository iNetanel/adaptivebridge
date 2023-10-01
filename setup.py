from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="adaptivebridge",
    version="0.9.0 alpha",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    author="Netanel Eliav",
    author_email="inetanel@me.com",
    url="https://github.com/inetanel/adaptivebridge",
    author_url="https://inetanel.com",
    description="Revolutionizing ML adaptive modelling for handling missing features and data. The model can predict and fill data gaps in real-world scenarios.",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',  # Specify Markdown format
    keywords=[
        'sklearn',
        'scikit-learn',
        'python',
        'data analysis',
        'machine learning',
        'data visualization',
        'python library',
        'data processing',
        'data science',
        'data exploration',
        'data manipulation',
        'analytics',
        'statistics',
        'artificial intelligence',
        'AI',
        'feature engineering',
        'data preprocessing',
        'predictive modeling',
        'classification',
        'regression',
        'missing data',
        'data cleaning',
        'data imputation',
        'data quality',
        'missing data analysis',
        'data handling',
        'data integrity',
        'data cleansing',
        'data wrangling',
        'data validation',
        'data completeness',
        'impute missing values',
        'data missingness',
        'missing data detection',
        'data quality assessment',
        'data pre-processing tool',
    ],
    
    license="MIT",

    project_urls={
    "Author Website": "https://inetanel.com",
    "Documentation": "https://inetanel.github.io/adaptivebridge",
    "Source Code": "https://github.com/inetanel/adaptivebridge",
    "Issue Tracker": "https://github.com/inetanel/adaptivebridge/issues",
                },
    classifiers=[
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
    "Intended Audience :: System Administrators",
    "Intended Audience :: Education",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
                ],
        )