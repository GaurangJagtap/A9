"""
Model Training Module.

Trains a Random Forest Classifier on the Breast Cancer dataset,
preserves feature names, and saves the trained model artifact to disk.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from src.data_loader import get_stratified_split


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
    max_depth: Optional[int] = None
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier baseline model.

    Args:
        X_train: Training features (DataFrame with feature names intact).
        y_train: Training target labels.
        n_estimators: Number of decision trees.
        random_state: Random state for reproducible training.
        max_depth: Maximum tree depth.

    Returns:
        Fitted RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth
    )
    rf.fit(X_train, y_train)
    return rf


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """Save trained model using joblib."""
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    print(f"Model saved successfully to {dest}")


def load_saved_model(filepath: str) -> RandomForestClassifier:
    """Load model from file."""
    src = Path(filepath)
    if not src.exists():
        raise FileNotFoundError(f"Model file not found: {src}")
    return joblib.load(src)


def run_pipeline() -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Execute training pipeline and save model artifact."""
    X_train, X_test, y_train, y_test, meta = get_stratified_split()
    model = train_model(X_train, y_train)
    
    # Save artifacts
    save_model(model, "models/random_forest_model.joblib")
    
    # Also save train/test splits for reuse across modules
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(data_dir / "X_train.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_train.to_csv(data_dir / "y_train.csv", index=False)
    y_test.to_csv(data_dir / "y_test.csv", index=False)
    
    print("Training pipeline finished. Data & model saved.")
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_pipeline()
