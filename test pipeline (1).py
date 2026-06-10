"""
test_pipeline.py
----------------
Basic unit tests to verify the pipeline components work correctly.
Run with: python -m pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train import train_adaboost, train_majority_voting


def make_dummy_data(n=500, fraud_ratio=0.5):
    """Generate a small balanced dummy dataset for testing."""
    np.random.seed(42)
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud

    X_legit = np.random.randn(n_legit, 30)
    X_fraud = np.random.randn(n_fraud, 30) + 1.5  # Slightly shifted

    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud)

    return X, y


def test_adaboost_trains_and_predicts():
    X, y = make_dummy_data()
    model = train_adaboost(X, y, n_estimators=10)
    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})
    print("✅ AdaBoost train+predict test passed")


def test_majority_voting_trains_and_predicts():
    X, y = make_dummy_data()
    model = train_majority_voting(X, y)
    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})
    print("✅ Majority Voting train+predict test passed")


def test_models_give_valid_accuracy():
    X, y = make_dummy_data(n=300)
    X_train, X_test = X[:250], X[250:]
    y_train, y_test = y[:250], y[250:]

    ada = train_adaboost(X_train, y_train, n_estimators=10)
    acc = (ada.predict(X_test) == y_test).mean()
    assert acc > 0.5, f"AdaBoost accuracy too low: {acc:.2f}"
    print(f"✅ AdaBoost accuracy on dummy data: {acc:.2f}")


if __name__ == "__main__":
    test_adaboost_trains_and_predicts()
    test_majority_voting_trains_and_predicts()
    test_models_give_valid_accuracy()
    print("\n✅ All tests passed!")
