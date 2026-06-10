"""
main.py
-------
Full pipeline: Load → Preprocess → Train → Evaluate → Save

Run with:
    python main.py
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import load_data, preprocess
from src.train import train_adaboost, train_majority_voting, save_model
from src.evaluate import evaluate_all

DATA_PATH = os.path.join("data", "creditcard.csv")


def main():
    print("=" * 60)
    print("  Credit Card Fraud Detection — Full Pipeline")
    print("  AdaBoost + Majority Voting | B.Tech Major Project")
    print("=" * 60)

    # ── Step 1: Load Data ──────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Dataset not found at: {DATA_PATH}")
        print("  Please download creditcard.csv from Kaggle and place it in the data/ folder.")
        print("  URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        sys.exit(1)

    df = load_data(DATA_PATH)

    # ── Step 2: Preprocess ─────────────────────────────────────
    X_train, X_test, y_train, y_test = preprocess(df)

    # ── Step 3: Train Models ───────────────────────────────────
    adaboost_model = train_adaboost(X_train, y_train)
    voting_model   = train_majority_voting(X_train, y_train)

    # ── Step 4: Evaluate ───────────────────────────────────────
    models = {
        "AdaBoost":        adaboost_model,
        "Majority Voting": voting_model
    }
    evaluate_all(models, X_test, y_test)

    # ── Step 5: Save Models ────────────────────────────────────
    save_model(adaboost_model, "adaboost")
    save_model(voting_model,   "majority_voting")

    print("\n" + "=" * 60)
    print("  ✅  Pipeline complete!")
    print("  📊  Reports saved in: reports/figures/")
    print("  💾  Models saved in:  models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
