"""
predict.py
----------
Load a saved model and predict on new transaction data.
Can be used for single rows or a CSV file.
"""

import pandas as pd
import numpy as np
from src.train import load_model
from sklearn.preprocessing import StandardScaler


def predict_from_csv(csv_path: str, model_name: str = "majority_voting"):
    """
    Predict fraud on a new CSV file (same format as creditcard.csv, without 'Class' column).
    """
    model = load_model(model_name)

    df = pd.read_csv(csv_path)

    # Scale Amount and Time if present
    scaler = StandardScaler()
    if 'Amount' in df.columns:
        df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df.drop('Amount', axis=1, inplace=True)
    if 'Time' in df.columns:
        df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        df.drop('Time', axis=1, inplace=True)
    if 'Class' in df.columns:
        df.drop('Class', axis=1, inplace=True)

    predictions = model.predict(df)
    df['Predicted_Class'] = predictions
    df['Result'] = df['Predicted_Class'].map({0: 'Legitimate', 1: 'FRAUD'})

    print(df[['Predicted_Class', 'Result']].value_counts())
    return df


def predict_single(transaction: dict, model_name: str = "majority_voting"):
    """
    Predict for a single transaction (as a dictionary of feature values).
    """
    model = load_model(model_name)
    row = pd.DataFrame([transaction])

    scaler = StandardScaler()
    if 'Amount' in row.columns:
        row['scaled_amount'] = scaler.fit_transform(row[['Amount']])
        row.drop('Amount', axis=1, inplace=True)
    if 'Time' in row.columns:
        row['scaled_time'] = scaler.fit_transform(row[['Time']])
        row.drop('Time', axis=1, inplace=True)

    pred = model.predict(row)[0]
    label = "⚠️  FRAUD DETECTED" if pred == 1 else "✅ Legitimate Transaction"
    print(f"\nPrediction: {label}")
    return pred
