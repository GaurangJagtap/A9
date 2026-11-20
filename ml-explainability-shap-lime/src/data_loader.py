"""
Data Loader Module for Breast Cancer Wisconsin Dataset.

Loads the dataset, inspects characteristics, performs sanity checks
(missing values, duplicates), and splits into stratified train/test sets.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Load Breast Cancer Wisconsin dataset from scikit-learn.

    Returns:
        X (pd.DataFrame): Feature matrix with original feature names.
        y (pd.Series): Binary target labels (0 = Malignant, 1 = Benign).
        metadata (dict): Dataset metadata including class names, descriptions, and shapes.
    """
    raw_data = load_breast_cancer(as_frame=True)
    X = raw_data.data.copy()
    y = raw_data.target.copy()
    
    # Target mapping explanation:
    # 0 -> Malignant (cancerous / harmful)
    # 1 -> Benign (non-cancerous / harmless)
    target_names = list(raw_data.target_names)  # ['malignant', 'benign']
    
    metadata = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target_names": target_names,
        "class_mapping": {0: "Malignant", 1: "Benign"},
        "missing_values": int(X.isnull().sum().sum()),
        "duplicates": int(X.duplicated().sum()),
        "class_distribution": y.value_counts().to_dict(),
        "limitations": (
            "1. Educational demonstration dataset only; not for clinical diagnosis.\n"
            "2. Small sample size (569 records) collected from FNA biopsies in 1995.\n"
            "3. High collinearity among radius, perimeter, and area features.\n"
            "4. Moderate class imbalance (~63% benign, ~37% malignant)."
        )
    }
    return X, y, metadata


def get_stratified_split(
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Loads and splits dataset into stratified train and test partitions.
    Preserves pandas DataFrame and Series structures to retain feature names.

    Args:
        test_size (float): Proportion of dataset for test set (default 0.20).
        random_state (int): Random seed for reproducibility (default 42).

    Returns:
        X_train, X_test, y_train, y_test, metadata
    """
    X, y, metadata = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, metadata


if __name__ == "__main__":
    X, y, meta = load_dataset()
    print("=== Dataset Summary ===")
    print(f"Total Samples: {meta['n_samples']}")
    print(f"Total Features: {meta['n_features']}")
    print(f"Target Distribution: {meta['class_distribution']}")
    print(f"Missing Values: {meta['missing_values']}")
    print(f"Duplicates: {meta['duplicates']}")
    print(f"Class Names: {meta['target_names']}")
