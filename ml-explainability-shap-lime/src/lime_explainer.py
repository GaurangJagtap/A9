"""
LIME Explainability Module.

Uses LimeTabularExplainer to produce local surrogate model explanations
for individual predictions on tabular data.
Generates explanation bar plots, HTML reports, and tabular summaries.
"""

from pathlib import Path
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import get_stratified_split
from src.train import load_saved_model


def get_lime_explainer(
    X_train: pd.DataFrame,
    class_names: List[str] = None,
    random_state: int = 42
) -> LimeTabularExplainer:
    """
    Initialize LimeTabularExplainer using the training dataset.

    Args:
        X_train: Training features DataFrame.
        class_names: List of class labels (e.g. ['Malignant', 'Benign']).
        random_state: Seed for perturbation sampling.

    Returns:
        Configured LimeTabularExplainer instance.
    """
    if class_names is None:
        class_names = ["Malignant", "Benign"]

    explainer = LimeTabularExplainer(
        training_data=np.ascontiguousarray(X_train.values),
        feature_names=list(X_train.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=random_state
    )
    return explainer


def explain_sample_lime(
    explainer: LimeTabularExplainer,
    model: RandomForestClassifier,
    sample_row: pd.Series,
    num_features: int = 10,
    predict_fn=None
) -> Any:
    """
    Generate LIME explanation for a single instance.

    Args:
        explainer: LimeTabularExplainer.
        model: Trained classifier.
        sample_row: Pandas Series representing the sample.
        num_features: Top N features to include in explanation.
        predict_fn: Prediction probability function (default: model.predict_proba).

    Returns:
        LIME Explanation object.
    """
    if predict_fn is None:
        predict_fn = model.predict_proba

    # Convert row to 1D numpy array float values
    data_row = np.ascontiguousarray(sample_row.values, dtype=np.float64)
    exp = explainer.explain_instance(
        data_row=data_row,
        predict_fn=predict_fn,
        num_features=num_features
    )
    return exp


def save_lime_plot(
    exp: Any,
    output_filepath: str,
    title: str = "LIME Local Explanation"
) -> None:
    """Save LIME explanation bar plot to file."""
    dest = Path(output_filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)

    fig = exp.as_pyplot_figure()
    plt.title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(dest, dpi=300)
    plt.close(fig)


def generate_lime_explanations(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "outputs/lime",
    sample_indices: Tuple[int, int] = (0, 1),
    num_features: int = 10
) -> List[dict]:
    """
    Generate LIME explanations for specified test samples and save visualizations.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    explainer = get_lime_explainer(X_train)
    results = []

    for idx in sample_indices:
        sample = X_test.iloc[idx]
        actual_label = int(y_test.iloc[idx])
        pred_label = int(model.predict(X_test.iloc[[idx]])[0])
        pred_proba = model.predict_proba(X_test.iloc[[idx]])[0]

        class_names = ["Malignant", "Benign"]
        exp = explain_sample_lime(
            explainer=explainer,
            model=model,
            sample_row=sample,
            num_features=num_features
        )

        # Save matplotlib figure
        plot_path = out_path / f"lime_sample_{idx}.png"
        title = (
            f"LIME Explanation for Test Sample Index {idx}\n"
            f"True: {class_names[actual_label]} | Pred: {class_names[pred_label]} "
            f"(P(Benign)={pred_proba[1]:.2f}, P(Malig)={pred_proba[0]:.2f})"
        )
        save_lime_plot(exp, str(plot_path), title=title)

        # Save HTML interactive report
        html_path = out_path / f"lime_sample_{idx}.html"
        exp.save_to_file(str(html_path))

        results.append({
            "sample_index": idx,
            "true_label": class_names[actual_label],
            "pred_label": class_names[pred_label],
            "proba_malignant": float(pred_proba[0]),
            "proba_benign": float(pred_proba[1]),
            "top_features": exp.as_list(),
            "plot_saved": str(plot_path)
        })

    print(f"LIME explanations generated and saved to {out_path.resolve()}")
    return results


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = get_stratified_split()
    model = load_saved_model("models/random_forest_model.joblib")
    generate_lime_explanations(model, X_train, X_test, y_test)
