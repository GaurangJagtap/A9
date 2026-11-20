"""
Unit and integration tests for ML Explainability pipeline.
Tests data loader, splitting, model training, predictions, probabilities,
SHAP computations, and LIME explanations.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_dataset, get_stratified_split
from src.train import train_model
from src.evaluate import evaluate_model
from src.shap_explainer import (
    get_shap_explainer,
    compute_shap_explanation,
    extract_class_shap_values
)
from src.lime_explainer import (
    get_lime_explainer,
    explain_sample_lime
)


@pytest.fixture(scope="module")
def data_splits():
    """Fixture providing train-test split data."""
    return get_stratified_split(test_size=0.20, random_state=42)


@pytest.fixture(scope="module")
def trained_rf_model(data_splits):
    """Fixture providing a fitted Random Forest model."""
    X_train, _, y_train, _, _ = data_splits
    model = train_model(X_train, y_train, n_estimators=50, random_state=42)
    return model


def test_dataset_loading():
    """Verify dataset loads properly with expected shapes and metadata."""
    X, y, meta = load_dataset()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (569, 30)
    assert len(y) == 569
    assert meta["missing_values"] == 0
    assert meta["duplicates"] == 0
    assert set(meta["target_names"]) == {"malignant", "benign"}


def test_stratified_split(data_splits):
    """Verify train-test splitting preserves stratifications and proportions."""
    X_train, X_test, y_train, y_test, meta = data_splits
    assert len(X_train) == 455
    assert len(X_test) == 114
    # Check feature count
    assert X_train.shape[1] == 30
    assert X_test.shape[1] == 30
    # Stratification check: ratio should be approximately preserved (~0.627 benign)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.02


def test_model_training_and_predictions(trained_rf_model, data_splits):
    """Verify model fits, generates predictions and valid probabilities."""
    _, X_test, _, y_test, _ = data_splits
    preds = trained_rf_model.predict(X_test)
    probas = trained_rf_model.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probas.shape == (len(X_test), 2)
    # Check probability sum equals 1
    assert np.allclose(probas.sum(axis=1), 1.0)
    # Check probabilities are between 0 and 1
    assert np.all((probas >= 0.0) & (probas <= 1.0))


def test_model_evaluation_metrics(trained_rf_model, data_splits, tmp_path):
    """Verify metrics calculation and format."""
    _, X_test, _, y_test, _ = data_splits
    metrics = evaluate_model(trained_rf_model, X_test, y_test, output_dir=str(tmp_path))

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] >= 0.90
    assert metrics["roc_auc"] >= 0.90


def test_shap_explanation_generation(trained_rf_model, data_splits):
    """Verify SHAP explainer runs and retains feature names and dimension."""
    X_train, X_test, _, _, _ = data_splits
    explainer = get_shap_explainer(trained_rf_model)
    explanation = compute_shap_explanation(explainer, X_test.iloc[:5])
    
    assert explanation is not None
    # Check feature name preservation
    class_exp = extract_class_shap_values(explanation, target_class=1)
    assert list(class_exp.feature_names) == list(X_test.columns)
    assert class_exp.values.shape == (5, 30)


def test_lime_explanation_generation(trained_rf_model, data_splits):
    """Verify LIME explainer runs, features match, and returns non-empty weights."""
    X_train, X_test, _, _, _ = data_splits
    explainer = get_lime_explainer(X_train)
    exp = explain_sample_lime(
        explainer=explainer,
        model=trained_rf_model,
        sample_row=X_test.iloc[0],
        num_features=5
    )
    feature_list = exp.as_list()
    assert len(feature_list) == 5
    for feat_condition, weight in feature_list:
        assert isinstance(feat_condition, str)
        assert isinstance(weight, float)


def test_invalid_input_handling(trained_rf_model):
    """Verify appropriate errors are raised on mismatched/corrupt input dimensions."""
    invalid_input = np.array([[1.0, 2.0, 3.0]])  # Only 3 features instead of 30
    with pytest.raises(ValueError):
        trained_rf_model.predict(invalid_input)
