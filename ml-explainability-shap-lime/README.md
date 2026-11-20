# Explainable AI (XAI): SHAP & LIME for Machine Learning Models

A complete, beginner-friendly, and educational Machine Learning project demonstrating model explainability using **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** on the Breast Cancer Wisconsin dataset.

---

> **⚠️ Disclaimer:** This project is strictly an educational demonstration of Machine Learning explainability principles for academic and viva presentation. It is **not** a certified clinical diagnostic software.

---

## 📌 Project Overview & Objectives

In high-stakes applications like healthcare, banking, and criminal justice, high-accuracy models ("black boxes") cannot be trusted without understanding **why** they make specific predictions.

This project demonstrates:
1. **Model Prediction:** What the trained model predicts for clinical tissue biopsy samples.
2. **Feature Influences:** Which cellular attributes (e.g., perimeter, concave points, texture) drive decisions.
3. **SHAP (Global & Local):** How Cooperative Game Theory rigorously attributes credit to features across the entire population and for individual patients.
4. **LIME (Local):** How fitting an interpretable surrogate linear model to local synthetic perturbations reveals threshold-based decision rules.
5. **SHAP vs LIME Comparison:** How their mathematical foundations, guarantees, and explanation styles differ.
6. **Interactive Exploration:** A lightweight local Streamlit web application for interactive viva demonstration.

---

## 📊 Dataset Details

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) via `sklearn.datasets.load_breast_cancer`.
- **Samples:** 569 instances.
- **Features:** 30 numeric continuous cellular nucleus features computed from digitized images of fine needle aspirate (FNA) biopsies.
- **Target Variable:** Binary diagnostic classification.
  - `0`: **Malignant** (harmful/cancerous) — 212 samples (~37.3%).
  - `1`: **Benign** (non-cancerous) — 357 samples (~62.7%).
- **Data Integrity:** 0 missing values, 0 duplicate records.
- **Train/Test Split:** 80% training (455 samples), 20% testing (114 samples), stratified by class label (`random_state=42`).
- **Limitations:** Small sample size, high inter-feature collinearity among geometric radius/area/perimeter metrics, and lack of diverse demographic metadata.

---

## 🤖 Machine Learning Model & Evaluation

A **Random Forest Classifier** (`n_estimators=100`, `random_state=42`) was trained on the 80% training partition, maintaining full feature name integrity.

### Actual Empirical Performance on Test Data (114 samples)

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **95.61%** | Overall correct classifications (109 / 114) |
| **Precision (Benign)** | **95.89%** | Accuracy of positive benign predictions |
| **Recall (Benign)** | **97.22%** | Proportion of actual benign cases identified |
| **F1-Score (Benign)** | **96.55%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **99.37%** | Separability across diagnostic classification thresholds |

### Classification Report

```text
              precision    recall  f1-score   support

   Malignant       0.95      0.93      0.94        42
      Benign       0.96      0.97      0.97        72

    accuracy                           0.96       114
   macro avg       0.95      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114
```

---

## 🧠 Explainability Methodologies

### 1. SHAP (SHapley Additive exPlanations)
- **Theoretical Basis:** Rooted in cooperative game theory (Lloyd Shapley, 1953). Treats each feature as a "player" in a coalition and assigns payout according to its marginal contribution across all feature subsets.
- **Axiomatic Guarantees:** Uniquely satisfies **Efficiency**, **Symmetry**, **Dummy (Null player)**, and **Additivity**.
- **Implementation:** Implemented using `shap.TreeExplainer` for exact and fast tree-based attribution ($O(TLD^2)$ complexity). Correctly handles 3-dimensional multi-class output slices for binary classification.
- **Generated Visualizations:**
  - `outputs/shap/shap_global_importance_bar.png`: Mean absolute SHAP value rankings across all test samples. Top global features: `worst concave points`, `worst perimeter`, `worst radius`, `mean concave points`.
  - `outputs/shap/shap_beeswarm_plot.png`: Distribution and directional influence of feature values (e.g., high `worst concave points` strongly pushes prediction towards Malignant).
  - `outputs/shap/shap_summary_plot.png`: Global distribution density plot.
  - `outputs/shap/shap_waterfall_sample_0.png` & `sample_1.png`: Local breakdown starting from baseline expected value $E[f(X)]$ to final output $f(x)$.

### 2. LIME (Local Interpretable Model-agnostic Explanations)
- **Theoretical Basis:** Introduced by Ribeiro et al. (2016). Assumes complex black-box decision surfaces can be approximated locally by a simple linear model.
- **Process:** Generates ~5000 synthetic perturbations around the target instance, weights them exponentially by Euclidean distance in feature space, obtains predictions from the black-box model, and fits a sparse weighted linear regression.
- **Implementation:** Configured via `LimeTabularExplainer` using training distribution statistics and continuous variable discretization into readable thresholds (e.g. `worst radius > 18.79`).
- **Generated Visualizations:**
  - `outputs/lime/lime_sample_0.png` & `sample_1.png`: Matplotlib horizontal explanation bar charts.
  - `outputs/lime/lime_sample_0.html` & `sample_1.html`: Standalone interactive HTML inspection reports.

---

## ⚖️ SHAP vs LIME Comparison

| Dimension | SHAP | LIME |
| :--- | :--- | :--- |
| **Core Idea** | Fair payout division from Cooperative Game Theory | Fit a simple linear surrogate on local perturbations |
| **Scope** | Both **Global** and **Local** explainability | Strictly **Local** explainability |
| **Mathematical Guarantees** | Provably satisfies 4 fundamental game-theoretic axioms | Heuristic approximation; no axiomatic guarantees |
| **Stability / Consistency** | Deterministic & consistent across runs | Can fluctuate slightly due to random perturbation sampling |
| **Rule Formats** | Additive numerical attribution weights | Discretized human-readable interval rules (`worst area <= 700`) |
| **Speed** | Extremely fast with `TreeExplainer`; slow with `KernelExplainer` | Moderate speed (~1-2 seconds per instance) |
| **Best Scenario** | Regulatory audits, feature attribution, model debugging | Intuitive rule-based spot-checks for end users |

---

## 📁 Project Directory Structure

```text
ml-explainability-shap-lime/
├── data/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── models/
│   └── random_forest_model.joblib
├── notebooks/
│   └── explainability_demo.ipynb
├── outputs/
│   ├── comparison/
│   │   ├── sample_0_comparison.json
│   │   ├── shap_vs_lime_comparison_matrix.csv
│   │   └── shap_vs_lime_side_by_side_sample_0.png
│   ├── evaluation/
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   ├── metrics.json
│   │   └── roc_curve.png
│   ├── lime/
│   │   ├── lime_sample_0.html
│   │   ├── lime_sample_0.png
│   │   ├── lime_sample_1.html
│   │   └── lime_sample_1.png
│   └── shap/
│       ├── shap_beeswarm_plot.png
│       ├── shap_global_importance_bar.png
│       ├── shap_summary_plot.png
│       ├── shap_waterfall_sample_0.png
│       └── shap_waterfall_sample_1.png
├── src/
│   ├── __init__.py
│   ├── compare_explainers.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── lime_explainer.py
│   ├── shap_explainer.py
│   └── train.py
├── tests/
│   └── test_pipeline.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Step-by-Step Windows Execution Guide

All commands run directly in Windows PowerShell or Command Prompt.

### 1. Create and Activate Virtual Environment

```powershell
# Navigate into the project directory
cd "c:\College\SEM 5\MLOPs\Assignment_9\ml-explainability-shap-lime"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
# (Or in CMD: venv\Scripts\activate.bat)
```

### 2. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts

Execute the end-to-end ML & Explainability pipeline:

```powershell
# 1. Train the Random Forest Model
python -m src.train

# 2. Evaluate Performance (Generates metrics, confusion matrix, ROC curve)
python -m src.evaluate

# 3. Generate SHAP Explanations (Global summary, beeswarm, waterfall plots)
python -m src.shap_explainer

# 4. Generate LIME Explanations (Local perturbation plots and HTML reports)
python -m src.lime_explainer

# 5. Generate SHAP vs LIME Comparative Analysis
python -m src.compare_explainers
```

### 4. Run the Interactive Streamlit Web App

Launch the local interactive dashboard:

```powershell
streamlit run app.py
```
> The interface will open automatically in your browser at `http://localhost:8501`. You can slide through test instances, inspect actual patient feature measurements, and view real-time dynamic SHAP waterfall and LIME surrogate explanations side-by-side.

### 5. Run Automated Tests

Execute the automated test suite with pytest:

```powershell
python -m pytest tests/ -v
```

---

## 🎯 Viva Q&A Quick Reference

1. **Why explainability instead of just accuracy?**
   - High accuracy does not guarantee fair or clinically valid reasoning. Explainability detects data leakage, builds stakeholder trust, and complies with safety regulations.
2. **What does a positive vs negative SHAP value mean?**
   - In our binary setup targeting **Benign (Class 1)**: a positive SHAP value pulls the prediction towards Benign; a negative value pulls it towards **Malignant (Class 0)**.
3. **Why do LIME explanations differ slightly between runs?**
   - LIME relies on random sampling of synthetic perturbations around the target point. Unless `random_state` is fixed, slight variance in the local regression weights can occur.
4. **Which is better: SHAP or LIME?**
   - Neither is universally better: SHAP is mathematically grounded and consistent (ideal for global audits and regulatory reporting), while LIME is model-agnostic, fast, and presents intuitive threshold conditions (ideal for quick local inspections).
