"""
SHAP vs LIME Comparison Module.

Conducts side-by-side empirical and theoretical comparison of SHAP and LIME
on identical test instances and global aggregation.
Saves comparison tables, markdown reports, and comparative visual plots.
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

from src.data_loader import get_stratified_split
from src.train import load_saved_model
from src.shap_explainer import get_shap_explainer, compute_shap_explanation, extract_class_shap_values
from src.lime_explainer import get_lime_explainer, explain_sample_lime


def run_comparative_analysis(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "outputs/comparison",
    sample_index: int = 0
) -> Dict[str, Any]:
    """
    Compare SHAP and LIME explanations for a specific test sample and overall.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. SHAP computation for sample
    shap_explainer = get_shap_explainer(model)
    shap_exp_all = compute_shap_explanation(shap_explainer, X_test.iloc[[sample_index]])
    shap_exp = extract_class_shap_values(shap_exp_all, target_class=1)  # Class 1 (Benign)
    
    shap_values = shap_exp.values[0]
    feature_names = list(X_test.columns)
    
    # Create top SHAP features DataFrame
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values,
        "abs_shap": np.abs(shap_values)
    }).sort_values("abs_shap", ascending=False).reset_index(drop=True)

    # 2. LIME computation for same sample
    lime_explainer = get_lime_explainer(X_train)
    lime_exp = explain_sample_lime(
        lime_explainer,
        model,
        X_test.iloc[sample_index],
        num_features=10
    )
    lime_list = lime_exp.as_list()
    lime_df = pd.DataFrame(lime_list, columns=["rule", "lime_weight"])

    # 3. Methodological Comparison Matrix
    comparison_matrix = [
        {
            "Dimension": "Theoretical Foundation",
            "SHAP (SHapley Additive exPlanations)": "Cooperative Game Theory (Shapley Values from 1953)",
            "LIME (Local Interpretable Model-agnostic Explanations)": "Local surrogate modeling (Sparse linear regression with perturbations)"
        },
        {
            "Dimension": "Global vs Local",
            "SHAP (SHapley Additive exPlanations)": "Both Global (aggregation of Shapley values) and Local",
            "LIME (Local Interpretable Model-agnostic Explanations)": "Strictly Local (explains one instance at a time)"
        },
        {
            "Dimension": "Axiomatic Guarantees",
            "SHAP (SHapley Additive exPlanations)": "Satisfies Efficiency, Symmetry, Dummy (Null), and Additivity",
            "LIME (Local Interpretable Model-agnostic Explanations)": "No formal theoretical axiomatic guarantees; heuristic"
        },
        {
            "Dimension": "Consistency",
            "SHAP (SHapley Additive exPlanations)": "Provably consistent across models and transformations",
            "LIME (Local Interpretable Model-agnostic Explanations)": "May vary due to random sampling perturbations"
        },
        {
            "Dimension": "Computational Complexity",
            "SHAP (SHapley Additive exPlanations)": "High for exact KernelSHAP; fast for TreeSHAP O(TLD^2)",
            "LIME (Local Interpretable Model-agnostic Explanations)": "Moderate, proportional to number of perturbation samples (e.g. 5000)"
        },
        {
            "Dimension": "Feature Independence Assumption",
            "SHAP (SHapley Additive exPlanations)": "Assumes feature independence unless conditional TreeSHAP is used",
            "LIME (Local Interpretable Model-agnostic Explanations)": "Perturbs features independently; can create out-of-distribution synthetic points"
        },
        {
            "Dimension": "Output Format",
            "SHAP (SHapley Additive exPlanations)": "Additive feature attribution values in log-odds or probability margin",
            "LIME (Local Interpretable Model-agnostic Explanations)": "Linear regression weights on binned/discretized interval rules"
        },
        {
            "Dimension": "Best Use Case",
            "SHAP (SHapley Additive exPlanations)": "Auditing overall model behavior, regulatory compliance, fair feature attribution",
            "LIME (Local Interpretable Model-agnostic Explanations)": "Quick intuitive inspection of individual black-box decisions with threshold rules"
        }
    ]

    comp_df = pd.DataFrame(comparison_matrix)
    comp_df.to_csv(out_path / "shap_vs_lime_comparison_matrix.csv", index=False)

    # 4. Save JSON summary of sample comparison
    sample_summary = {
        "sample_index": sample_index,
        "true_label": int(y_test.iloc[sample_index]),
        "predicted_label": int(model.predict(X_test.iloc[[sample_index]])[0]),
        "predicted_probabilities": model.predict_proba(X_test.iloc[[sample_index]])[0].tolist(),
        "top_shap_features": shap_df.head(10).to_dict(orient="records"),
        "top_lime_features": lime_df.to_dict(orient="records")
    }
    with open(out_path / f"sample_{sample_index}_comparison.json", "w", encoding="utf-8") as f:
        json.dump(sample_summary, f, indent=4)

    # 5. Visual side-by-side comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Top 8 SHAP features
    top_shap = shap_df.head(8).sort_values("shap_value", ascending=True)
    axes[0].barh(top_shap["feature"], top_shap["shap_value"], color=np.where(top_shap["shap_value"] >= 0, "#2ecc71", "#e74c3c"))
    axes[0].axvline(0, color="black", linestyle="--", alpha=0.7)
    axes[0].set_title(f"SHAP Attributions (Class: Benign)\nSample Index {sample_index}")
    axes[0].set_xlabel("SHAP Value (impact on model output)")

    # Top 8 LIME features
    lime_top = lime_df.head(8).iloc[::-1]  # reverse for horizontal bar
    axes[1].barh(lime_top["rule"], lime_top["lime_weight"], color=np.where(lime_top["lime_weight"] >= 0, "#2ecc71", "#e74c3c"))
    axes[1].axvline(0, color="black", linestyle="--", alpha=0.7)
    axes[1].set_title(f"LIME Rule Weights (Class: Benign)\nSample Index {sample_index}")
    axes[1].set_xlabel("Local Surrogate Regression Weight")

    plt.tight_layout()
    plt.savefig(out_path / f"shap_vs_lime_side_by_side_sample_{sample_index}.png", dpi=300)
    plt.close()

    print(f"Comparison completed. Outputs saved to {out_path.resolve()}")
    return sample_summary


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = get_stratified_split()
    model = load_saved_model("models/random_forest_model.joblib")
    run_comparative_analysis(model, X_train, X_test, y_test, sample_index=0)
