"""
evaluate.py
-----------
Evaluates trained models and generates:
- Classification report
- Confusion matrix (saved as image)
- ROC-AUC curve (saved as image)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)


def print_classification_report(model, X_test, y_test, model_name: str):
    """Print precision, recall, F1 for fraud detection."""
    y_pred = model.predict(X_test)
    print(f"\n{'='*55}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    return y_pred


def plot_confusion_matrix(model, X_test, y_test, model_name: str, save_dir: str = "reports/figures"):
    """Plot and save confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()

    filename = os.path.join(save_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[PLOT] Confusion matrix saved: {filename}")


def plot_roc_curve(models_dict: dict, X_test, y_test, save_dir: str = "reports/figures"):
    """
    Plot ROC curves for multiple models on the same graph.
    models_dict: { 'Model Name': model_object }
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (name, model) in enumerate(models_dict.items()):
        # Use predict_proba if available, else decision_function
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC-AUC Curve — Fraud Detection Models", fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    filename = os.path.join(save_dir, "roc_auc_comparison.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[PLOT] ROC-AUC curve saved: {filename}")


def evaluate_all(models_dict: dict, X_test, y_test):
    """Run full evaluation for all models."""
    for name, model in models_dict.items():
        print_classification_report(model, X_test, y_test, model_name=name)
        plot_confusion_matrix(model, X_test, y_test, model_name=name)

    plot_roc_curve(models_dict, X_test, y_test)
    print("\n[INFO] All evaluation reports generated in reports/figures/")
