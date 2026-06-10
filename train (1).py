"""
train.py
--------
Trains AdaBoost and Majority Voting ensemble models.
Saves trained models to the /models directory.
"""

import os
import joblib
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_adaboost(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Train an AdaBoost classifier with Decision Tree stumps as base estimators.
    """
    print("\n[TRAIN] Training AdaBoost classifier...")
    base_estimator = DecisionTreeClassifier(max_depth=1)  # Decision stump
    ada = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=1.0,
        random_state=random_state
    )
    ada.fit(X_train, y_train)
    print("[TRAIN] AdaBoost training complete.")
    return ada


def train_majority_voting(X_train, y_train, random_state: int = 42):
    """
    Train a Majority Voting ensemble combining:
    - Logistic Regression
    - Random Forest
    - AdaBoost
    Uses 'hard' voting (majority class wins).
    """
    print("\n[TRAIN] Training Majority Voting ensemble...")

    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        random_state=random_state
    )

    voting_clf = VotingClassifier(
        estimators=[
            ('logistic_regression', lr),
            ('random_forest', rf),
            ('adaboost', ada)
        ],
        voting='hard'
    )
    voting_clf.fit(X_train, y_train)
    print("[TRAIN] Majority Voting training complete.")
    return voting_clf


def save_model(model, model_name: str, save_dir: str = "models"):
    """Save a trained model to disk using joblib."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, path)
    print(f"[SAVE] Model saved to: {path}")
    return path


def load_model(model_name: str, save_dir: str = "models"):
    """Load a saved model from disk."""
    path = os.path.join(save_dir, f"{model_name}.pkl")
    model = joblib.load(path)
    print(f"[LOAD] Model loaded from: {path}")
    return model
