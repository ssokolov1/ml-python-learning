"""
Boston Housing Dataset Loader
=============================

This module provides a replacement for the removed sklearn.datasets.load_boston function.
It loads the Boston housing dataset for educational purposes.

Usage:
    from boston_loader import load_boston
    boston = load_boston()
"""

import pandas as pd
import numpy as np
from sklearn.utils import Bunch
import warnings


def load_boston():
    """
    Load Boston housing dataset for educational purposes.
    
    This recreates the functionality of the removed sklearn load_boston function.
    The dataset contains 506 samples with 13 features each, predicting house prices.
    
    Returns
    -------
    bunch : sklearn.utils.Bunch
        Dictionary-like object with the following attributes:
        - data : ndarray of shape (506, 13)
            The data matrix
        - target : ndarray of shape (506,)
            The regression target (house prices)
        - feature_names : ndarray of shape (13,)
            The names of the dataset features
        - DESCR : str
            The full description of the dataset
    
    Examples
    --------
    >>> boston = load_boston()
    >>> print(boston.data.shape)
    (506, 13)
    >>> print(boston.feature_names)
    ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
    """
    
    # Feature names for the Boston dataset
    feature_names = np.array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                             'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    
    # Dataset description
    description = """
Boston house prices dataset
---------------------------

**Data Set Characteristics:**

    :Number of Instances: 506
    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is the target.
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's (target)

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This dataset has been loaded for educational purposes. Please note that this dataset
contains a feature (B) that is problematic from an ethical standpoint.
"""
    
    try:
        # Try to load from the original CMU source
        data_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
        
        # Reshape the data according to the original format
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        
        print("Successfully loaded Boston dataset from original source")
        
    except Exception as e1:
        try:
            # Backup: Try alternative source
            print(f"Primary source failed ({e1}), trying alternative source...")
            alt_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
            df = pd.read_csv(alt_url)
            
            # Map columns to expected names if needed
            if 'medv' in df.columns:
                target = df['medv'].values
                data = df.drop('medv', axis=1).values
            else:
                # Assume last column is target
                data = df.iloc[:, :-1].values
                target = df.iloc[:, -1].values
            
            print("Successfully loaded Boston dataset from alternative source")
            
        except Exception as e2:
            # Final fallback: Create realistic synthetic data based on original statistics
            print(f"Alternative source also failed ({e2}), creating synthetic dataset...")
            print("Note: This is synthetic data with similar statistical properties to the original")
            
            np.random.seed(42)  # For reproducibility
            n_samples = 506
            
            # Create synthetic data with realistic ranges based on original dataset
            data = np.column_stack([
                np.random.exponential(3.6, n_samples),      # CRIM (0-89)
                np.random.uniform(0, 100, n_samples),       # ZN (0-100)
                np.random.uniform(0, 27, n_samples),        # INDUS (0-27)
                np.random.binomial(1, 0.07, n_samples),     # CHAS (0 or 1)
                np.random.uniform(0.385, 0.871, n_samples), # NOX (0.385-0.871)
                np.random.normal(6.3, 0.7, n_samples),      # RM (3.5-8.8)
                np.random.uniform(2.9, 100, n_samples),     # AGE (2.9-100)
                np.random.uniform(1.1, 12, n_samples),      # DIS (1.1-12)
                np.random.randint(1, 25, n_samples),        # RAD (1-24)
                np.random.uniform(187, 711, n_samples),     # TAX (187-711)
                np.random.uniform(12.6, 22, n_samples),     # PTRATIO (12.6-22)
                np.random.uniform(0.32, 396.9, n_samples), # B (0.32-396.9)
                np.random.uniform(1.73, 37.97, n_samples)  # LSTAT (1.73-37.97)
            ])
            
            # Create realistic target values
            target = (20 + 
                     data[:, 5] * 5 +          # Rooms effect
                     -data[:, 0] * 0.1 +       # Crime effect
                     -data[:, 12] * 0.5 +      # LSTAT effect
                     np.random.normal(0, 3, n_samples))  # Noise
            
            target = np.clip(target, 5, 50)  # Realistic price range
    
    # Ensure data has correct shape
    if data.shape[1] != 13:
        warnings.warn(f"Expected 13 features, got {data.shape[1]}. Adjusting dataset.")
        if data.shape[1] > 13:
            data = data[:, :13]
        else:
            # Pad with zeros if needed
            padding = np.zeros((data.shape[0], 13 - data.shape[1]))
            data = np.hstack([data, padding])
    
    return Bunch(
        data=data,
        target=target,
        feature_names=feature_names,
        DESCR=description,
        filename='boston_house_prices.csv'
    )


# For compatibility with mglearn
def load_extended_boston():
    """
    Load extended Boston dataset (same as regular Boston for this implementation).
    This function provides compatibility with mglearn.datasets.load_extended_boston()
    
    Returns
    -------
    X : ndarray of shape (506, 13)
        The data matrix
    y : ndarray of shape (506,)
        The target values
    """
    boston = load_boston()
    return boston.data, boston.target


if __name__ == "__main__":
    # Test the loader
    print("Testing Boston dataset loader...")
    boston = load_boston()
    print(f"Data shape: {boston.data.shape}")
    print(f"Target shape: {boston.target.shape}")
    print(f"Feature names: {boston.feature_names}")
    print(f"Sample data point: {boston.data[0]}")
    print(f"Sample target: {boston.target[0]}")
    
    # Test extended loader
    X, y = load_extended_boston()
    print(f"Extended loader - X shape: {X.shape}, y shape: {y.shape}")