"""
SHAP Explainability Module.

Uses SHAP TreeExplainer for Random Forest.
Computes SHAP values, generates:
1. Global Feature Importance Bar Plot.
2. SHAP Summary Plot.
3. SHAP Beeswarm Plot.
4. Individual Waterfall Plots for sample explanation.
Handles binary classification dimensions properly.
"""

from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import get_stratified_split
from src.train import load_saved_model


def get_shap_explainer(model: RandomForestClassifier) -> shap.TreeExplainer:
    """Initialize and return SHAP TreeExplainer."""
    return shap.TreeExplainer(model)


def compute_shap_explanation(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame
) -> shap.Explanation:
    """
    Compute SHAP values for given dataset X.
    Returns shap.Explanation object.
    """
    return explainer(X)


def extract_class_shap_values(
    explanation: shap.Explanation,
    target_class: int = 1
) -> shap.Explanation:
    """
    Safely extract SHAP Explanation object for a specific target class.
    
    In binary classification with TreeExplainer:
    - explanation.values has shape (N, num_features, 2) or (N, num_features)
    - If 3D, slice along the last dimension for target_class.
    """
    if len(explanation.shape) == 3 and explanation.shape[2] == 2:
        class_exp = shap.Explanation(
            values=explanation.values[:, :, target_class],
            base_values=explanation.base_values[:, target_class]
            if len(explanation.base_values.shape) > 1
            else explanation.base_values,
            data=explanation.data,
            feature_names=explanation.feature_names
        )
        return class_exp
    return explanation


def generate_shap_plots(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str = "outputs/shap",
    sample_indices: Tuple[int, int] = (0, 1),
    target_class: int = 1
) -> dict:
    """
    Generate and save all required SHAP plots.

    Args:
        model: Trained classifier.
        X_train: Training features (for reference/background).
        X_test: Test features to explain.
        output_dir: Directory to save generated figures.
        sample_indices: Indices of test samples for local waterfall plots.
        target_class: 1 for Benign (or 0 for Malignant).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    explainer = get_shap_explainer(model)
    explanation = compute_shap_explanation(explainer, X_test)
    class_exp = extract_class_shap_values(explanation, target_class=target_class)

    target_name = "Benign (Class 1)" if target_class == 1 else "Malignant (Class 0)"

    # 1. Global Feature Importance Bar Plot
    plt.figure(figsize=(9, 7))
    shap.plots.bar(class_exp, max_display=15, show=False)
    plt.title(f"SHAP Global Feature Importance (Mean |SHAP Value|) - {target_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path / "shap_global_importance_bar.png", dpi=300)
    plt.close()

    # 2. SHAP Beeswarm Plot
    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(class_exp, max_display=15, show=False)
    plt.title(f"SHAP Beeswarm Summary Plot - {target_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path / "shap_beeswarm_plot.png", dpi=300)
    plt.close()

    # 3. SHAP Summary Plot (Standard Dot / Violin)
    plt.figure(figsize=(9, 7))
    # We pass values and data with feature names
    shap.summary_plot(
        class_exp.values,
        features=X_test,
        feature_names=list(X_test.columns),
        max_display=15,
        show=False
    )
    plt.title(f"SHAP Summary Distribution Plot - {target_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path / "shap_summary_plot.png", dpi=300)
    plt.close()

    # 4. Local Individual Explanations (Waterfall Plots)
    for idx in sample_indices:
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(class_exp[idx], max_display=12, show=False)
        plt.title(f"SHAP Waterfall Explanation - Test Sample Index {idx} ({target_name})", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path / f"shap_waterfall_sample_{idx}.png", dpi=300)
        plt.close()

    print(f"SHAP plots generated and saved to {out_path.resolve()}")
    return {
        "explainer": explainer,
        "explanation": class_exp
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = get_stratified_split()
    model = load_saved_model("models/random_forest_model.joblib")
    generate_shap_plots(model, X_train, X_test)
