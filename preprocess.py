"""
preprocess.py
-------------
Handles data loading, cleaning, scaling, and SMOTE oversampling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Kaggle credit card fraud CSV."""
    print(f"[INFO] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Class distribution:\n{df['Class'].value_counts()}")
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the dataset:
    - Scale 'Amount' and 'Time' features
    - Split into train/test
    - Apply SMOTE only on training data to handle class imbalance

    Returns:
        X_train, X_test, y_train, y_test (after SMOTE on train)
    """
    # Scale Amount and Time (V1-V28 are already PCA-transformed)
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train/test split — stratified to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\n[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"[INFO] Fraud cases in test set: {y_test.sum()}")

    # Apply SMOTE on training data only
    print("\n[INFO] Applying SMOTE to balance training data...")
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE — Train size: {X_train_res.shape[0]}")
    print(f"[INFO] After SMOTE — Class distribution: {pd.Series(y_train_res).value_counts().to_dict()}")

    return X_train_res, X_test, y_train_res, y_test
