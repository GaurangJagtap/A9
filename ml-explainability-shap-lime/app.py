"""
Interactive Streamlit Application for Model Predictions & Explainability (SHAP & LIME).

Provides:
- Test sample selection and inspection.
- Real-time model prediction and probabilities.
- Dynamic SHAP waterfall calculation and visualization.
- Dynamic LIME local perturbation calculation and visualization.
- Educational explainability side-by-side comparison notes for viva demonstration.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import shap

# Ensure src can be imported
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import get_stratified_split
from src.train import load_saved_model
from src.shap_explainer import get_shap_explainer, compute_shap_explanation, extract_class_shap_values
from src.lime_explainer import get_lime_explainer, explain_sample_lime


st.set_page_config(
    page_title="ML Explainability: SHAP vs LIME",
    page_icon="🔬",
    layout="wide"
)


@st.cache_resource
def load_cached_data_and_model():
    """Cache dataset splits and trained model."""
    X_train, X_test, y_train, y_test, meta = get_stratified_split()
    model = load_saved_model("models/random_forest_model.joblib")
    lime_explainer = get_lime_explainer(X_train)
    shap_explainer = get_shap_explainer(model)
    return X_train, X_test, y_train, y_test, meta, model, lime_explainer, shap_explainer


X_train, X_test, y_train, y_test, meta, model, lime_explainer, shap_explainer = load_cached_data_and_model()

# Header
st.title("🔬 ML Explainability Demo: SHAP & LIME")
st.markdown(
    """
    **Project Objective:** Educational demonstration of Black-Box Machine Learning Explainability 
    using **SHAP** (Game Theory Shapley values) and **LIME** (Local Surrogate Models) on the 
    Breast Cancer Wisconsin Dataset.
    """
)

# Sidebar configuration
st.sidebar.header("🕹️ Sample Selection")
sample_idx = st.sidebar.slider(
    "Select Test Sample Index:",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Target Mapping:**
    - **0**: Malignant (Cancerous)
    - **1**: Benign (Non-Cancerous)
    
    *Educational demonstration only. Not for clinical diagnostic use.*
    """
)

# Extract chosen sample
sample_row = X_test.iloc[sample_idx]
true_label = int(y_test.iloc[sample_idx])
true_name = "Benign (1)" if true_label == 1 else "Malignant (0)"

# Inference
pred_class = int(model.predict(X_test.iloc[[sample_idx]])[0])
pred_name = "Benign (1)" if pred_class == 1 else "Malignant (0)"
pred_proba = model.predict_proba(X_test.iloc[[sample_idx]])[0]

# Display Prediction Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Test Sample Index", f"#{sample_idx}")
with col2:
    st.metric("Ground Truth Label", true_name)
with col3:
    is_correct = "✅" if pred_class == true_label else "❌"
    st.metric("Model Prediction", f"{pred_name} {is_correct}")
with col4:
    st.metric("Confidence (Benign)", f"{pred_proba[1]*100:.1f}%")

st.markdown("---")

# Feature Inspection
with st.expander("🔍 View Selected Sample Feature Values (All 30 Features)", expanded=False):
    st.dataframe(pd.DataFrame(sample_row).T, use_container_width=True)

# Tabs for Explainability
tab_shap, tab_lime, tab_compare = st.tabs([
    "📈 SHAP (Shapley Additive exPlanations)",
    "🍋 LIME (Local Interpretable Model-agnostic Explanations)",
    "⚖️ Side-by-Side Comparison & Theory"
])

# ----------------- SHAP TAB -----------------
with tab_shap:
    st.subheader(f"SHAP Waterfall Explanation for Sample #{sample_idx}")
    st.markdown(
        """
        - **$E[f(X)]$ (Base Value):** The average model prediction over the background training dataset.
        - **$f(x)$:** The predicted probability for this instance.
        - **Red arrows (+):** Push prediction higher toward **Benign (Class 1)**.
        - **Blue arrows (-):** Push prediction lower toward **Malignant (Class 0)**.
        """
    )
    
    with st.spinner("Computing SHAP values..."):
        shap_raw = compute_shap_explanation(shap_explainer, X_test.iloc[[sample_idx]])
        shap_exp = extract_class_shap_values(shap_raw, target_class=1)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_exp[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Top Contributing Features (SHAP)")
    shap_vals = shap_exp.values[0]
    top_shap_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Actual Value": sample_row.values,
        "SHAP Attribution": shap_vals,
        "Absolute Impact": np.abs(shap_vals)
    }).sort_values("Absolute Impact", ascending=False).head(8)
    st.dataframe(top_shap_df[["Feature", "Actual Value", "SHAP Attribution"]], use_container_width=True)


# ----------------- LIME TAB -----------------
with tab_lime:
    st.subheader(f"LIME Local Explanation for Sample #{sample_idx}")
    st.markdown(
        """
        - LIME creates synthetic perturbations around this instance and fits an interpretable 
          sparse linear surrogate model weighted by distance.
        - Green features support Benign; Orange/Red features support Malignant.
        """
    )

    with st.spinner("Fitting local surrogate model via LIME..."):
        lime_exp = explain_sample_lime(
            explainer=lime_explainer,
            model=model,
            sample_row=sample_row,
            num_features=10
        )
        
        lime_fig = lime_exp.as_pyplot_figure()
        plt.tight_layout()
        st.pyplot(lime_fig)
        plt.close(lime_fig)

    st.markdown("#### LIME Rule Weights Table")
    lime_rules = lime_exp.as_list()
    lime_table = pd.DataFrame(lime_rules, columns=["Discretized Feature Rule", "Surrogate Regression Weight"])
    st.dataframe(lime_table, use_container_width=True)


# ----------------- COMPARISON TAB -----------------
with tab_compare:
    st.subheader("⚖️ SHAP vs LIME: Theoretical & Empirical Comparison")
    
    st.markdown(
        """
        | Dimension | SHAP | LIME |
        | :--- | :--- | :--- |
        | **Foundational Theory** | Cooperative Game Theory (Shapley values, 1953) | Local Surrogate Modeling (Sparse linear model) |
        | **Axiomatic Guarantees**| Efficiency, Symmetry, Dummy, Additivity | No formal mathematical guarantees (Heuristic) |
        | **Global Explainability**| Yes (Averages individual Shapley attributions globally) | No (Strictly local to each sample) |
        | **Consistency** | Mathematically guaranteed consistent | May fluctuate between runs due to random perturbations |
        | **Feature Rules** | Continuous numeric feature attribution | Binned / discretized threshold conditions (e.g. `worst radius > 18.79`) |
        | **Speed** | Extremely fast with TreeSHAP ($O(TLD^2)$) | Moderate (requires simulating ~5000 perturbations) |
        """
    )
    
    st.markdown("### Why Explainability Matters in High-Stakes ML")
    st.markdown(
        """
        1. **Detecting Model Bias / Data Leakage:** Ensures models make predictions for legitimate clinical/biomarker reasons rather than spurious artifacts.
        2. **Trust & Human-in-the-Loop:** Clinical practitioners need to verify *why* a specimen is classified as malignant before taking action.
        3. **Regulatory Compliance:** Essential for FDA, GDPR ("Right to Explanation"), and AI Safety frameworks.
        """
    )

st.markdown("---")
st.caption("College ML Assignment — Machine Learning Explainability Project (SHAP & LIME)")
