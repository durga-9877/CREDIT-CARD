from setuptools import setup, find_packages

setup(
    name="credit-card-fraud-detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask>=2.0.1',
        'pandas>=1.5.3',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.2',
        'imbalanced-learn>=0.9.0',
        'joblib>=1.1.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.2',
    ],
) 