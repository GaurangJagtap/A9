"""
Model Evaluation Module.

Evaluates the trained Random Forest Classifier on test data.
Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Generates and saves Confusion Matrix, ROC curve, and classification report.
"""

import json
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import get_stratified_split
from src.train import load_saved_model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "outputs/evaluation"
) -> Dict[str, Any]:
    """
    Perform complete model evaluation on test data.

    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Ground truth labels.
        output_dir: Directory where plots and JSON metrics will be saved.

    Returns:
        metrics (dict): Numeric evaluation metrics and classification report.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (Benign)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=["Malignant", "Benign"],
        output_dict=True
    )
    clf_report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Malignant", "Benign"]
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report
    }

    # Save metrics JSON
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save Classification Report as text
    with open(out_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(clf_report_text)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"]
    )
    plt.title("Confusion Matrix - Random Forest Classifier")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrix.png", dpi=300)
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / "roc_curve.png", dpi=300)
    plt.close()

    print(f"Evaluation completed. Metrics & plots saved to {out_path.resolve()}")
    print("\n--- Model Evaluation Summary ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    return metrics


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = get_stratified_split()
    model = load_saved_model("models/random_forest_model.joblib")
    evaluate_model(model, X_test, y_test)
