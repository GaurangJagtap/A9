import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import os

doc = docx.Document()

# Set standard margins (1 inch)
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_hex)
    tcPr.append(shd)

def add_styled_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    run = h.runs[0]
    run.font.name = 'Calibri'
    if level == 1:
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0, 51, 102)
    elif level == 2:
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0, 76, 153)
    return h

# Header Title Block
if os.path.exists('college_logo.png'):
    logo_p = doc.add_paragraph()
    logo_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    logo_p.paragraph_format.space_after = Pt(4)
    logo_run = logo_p.add_run()
    logo_run.add_picture('college_logo.png', width=Inches(1.2))

p_inst = doc.add_paragraph()
p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_inst.paragraph_format.space_after = Pt(2)
run_inst = p_inst.add_run("Bansilal Ramnath Agarwal Charitable Trust's\nVishwakarma Institute of Technology, Pune\n")
run_inst.font.name = 'Calibri'
run_inst.font.size = Pt(13)
run_inst.font.bold = True
run_inst.font.color.rgb = RGBColor(0, 51, 102)

p_dept = doc.add_paragraph()
p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_dept.paragraph_format.space_after = Pt(8)
run_dept = p_dept.add_run("Department of Artificial Intelligence and Data Science\n")
run_dept.font.name = 'Calibri'
run_dept.font.size = Pt(11)
run_dept.font.bold = True

p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_after = Pt(12)
run_title = p_title.add_run("Assignment No. 9: Apply SHAP and LIME for Model Explainability")
run_title.font.name = 'Calibri'
run_title.font.size = Pt(13.5)
run_title.font.bold = True
run_title.font.color.rgb = RGBColor(180, 0, 0)

# Cover Table
meta_data = [
    ("Department", "Artificial Intelligence and Data Science"),
    ("Student Name", "[Full Name]"),
    ("Class", "TY"),
    ("Division", "B"),
    ("Roll No", "[Roll Number]"),
    ("PRN No", "{PRN Number}"),
    ("Semester", "SEM V"),
    ("Academic Year", "2026-2027"),
    ("Subject Name & Code", "MLOps (AI3206C)"),
    ("Assignment No.", "Assignment No. 9"),
    ("Title of Assignment", "Apply SHAP/LIME for Explainability — Complete ML Project"),
    ("Date of Performance / Submission", "21/11/2026 / 21/11/2026")
]

table = doc.add_table(rows=len(meta_data), cols=2)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = False

for i, (k, v) in enumerate(meta_data):
    row = table.rows[i]
    cell_k, cell_v = row.cells[0], row.cells[1]
    cell_k.width = Inches(2.5)
    cell_v.width = Inches(4.0)
    
    cell_k.text = k
    cell_v.text = v
    
    run_k = cell_k.paragraphs[0].runs[0]
    run_k.font.name = 'Calibri'
    run_k.font.size = Pt(10)
    run_k.font.bold = True
    
    run_v = cell_v.paragraphs[0].runs[0]
    run_v.font.name = 'Calibri'
    run_v.font.size = Pt(10)
    
    set_cell_background(cell_k, 'F2F4F7')
    set_cell_background(cell_v, 'FFFFFF' if i % 2 == 0 else 'FAFAFA')

doc.add_page_break()

# Aim
add_styled_heading(doc, 'Aim', 1)
p_aim = doc.add_paragraph('To design, implement, evaluate, and demonstrate an end-to-end Explainable Machine Learning (XAI) pipeline using Scikit-Learn, SHAP (SHapley Additive exPlanations), and LIME (Local Interpretable Model-agnostic Explanations) on the Breast Cancer Wisconsin dataset, accompanied by an interactive Streamlit local explanation interface.')
p_aim.paragraph_format.line_spacing = 1.15
p_aim.paragraph_format.space_after = Pt(8)

# Problem Statement
add_styled_heading(doc, 'Problem Statement', 1)
p_ps = doc.add_paragraph('Modern high-capacity Machine Learning models (such as Random Forests, Gradient Boosted Trees, and Deep Neural Networks) operate as opaque black boxes. While achieving high predictive accuracy, their internal reasoning mechanisms are inaccessible to end users and domain practitioners. In high-stakes domains like clinical oncology, deployment without transparency leads to trust deficits, inability to detect data leakage or systemic bias, and non-compliance with regulatory standards (such as GDPR Right to Explanation and FDA AI/ML guidelines). Therefore, this project implements both cooperative game-theoretic attribution (SHAP) and local surrogate perturbation modeling (LIME) on a 30-feature clinical biopsy classification task to systematically explain predictions globally and locally.')
p_ps.paragraph_format.line_spacing = 1.15
p_ps.paragraph_format.space_after = Pt(8)

# Objectives
add_styled_heading(doc, 'Objectives', 1)
objs = [
    'CO1: Understand model interpretability and explainability within the production MLOps lifecycle.',
    'CO2: Implement data preprocessing and train a baseline Random Forest Classifier with intact feature identity.',
    'CO3: Implement SHAP TreeExplainer to compute exact Shapley attributions, generating global beeswarm, summary, and local waterfall plots.',
    'CO4: Implement LIME LimeTabularExplainer to approximate local decision boundaries via synthetic perturbation surrogates and threshold rules.',
    'CO5: Perform empirical and theoretical comparative analysis between SHAP and LIME across mathematical axioms, stability, and speed.',
    'CO6: Build a modular, test-driven Python application with an interactive Streamlit demonstration dashboard.'
]
for obj in objs:
    p = doc.add_paragraph(obj, style='List Bullet')
    p.paragraph_format.space_after = Pt(3)

# Tools, Technologies and Environment
add_styled_heading(doc, 'Tools, Technologies and Environment', 1)
env_data = [
    ('Operating System', 'Windows 11 / Windows PowerShell'),
    ('Programming Language', 'Python 3.12.10'),
    ('Machine Learning Framework', 'Scikit-Learn 1.9.1, NumPy 2.5.4, Pandas 2.2.3, Joblib 1.6.0'),
    ('Explainability Libraries', 'SHAP 0.54.1 (TreeExplainer), LIME 0.2.0.1 (LimeTabularExplainer)'),
    ('Visualization Frameworks', 'Matplotlib 3.11.2, Seaborn 0.13.2'),
    ('Interactive User Interface', 'Streamlit 1.67.0'),
    ('Testing & Quality Assurance', 'Pytest 9.2.0 (7 Unit and Integration Tests Passed)'),
    ('Dataset / Model', 'Breast Cancer Wisconsin (Diagnostic), 569 instances, 30 continuous features')
]
env_table = doc.add_table(rows=len(env_data)+1, cols=2)
env_table.alignment = WD_TABLE_ALIGNMENT.CENTER
header_cells = env_table.rows[0].cells
header_cells[0].text = 'Item'
header_cells[1].text = 'Details'
set_cell_background(header_cells[0], '003366')
set_cell_background(header_cells[1], '003366')
for c in header_cells:
    for run in c.paragraphs[0].runs:
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)

for i, (k, v) in enumerate(env_data):
    row = env_table.rows[i+1]
    row.cells[0].text = k
    row.cells[1].text = v
    set_cell_background(row.cells[0], 'F2F4F7')
    set_cell_background(row.cells[1], 'FFFFFF' if i % 2 == 0 else 'FAFAFA')

doc.add_page_break()

# Background Theory
add_styled_heading(doc, 'Background Theory', 1)

add_styled_heading(doc, 'a) Introduction to Machine Learning Explainability (XAI)', 2)
p_xai = doc.add_paragraph('Explainable Artificial Intelligence (XAI) encompasses a suite of processes and methods that allow human users to comprehend and trust the outputs created by machine learning algorithms. In traditional ML workflows, models were evaluated exclusively on scalar predictive metrics (Accuracy, ROC-AUC, F1-Score). However, a model can achieve high accuracy by latching onto spurious correlations or hospital-specific demographic artifacts rather than genuine pathological biomarkers. Explainability bridges the gap between high accuracy and clinical safety by providing actionable insights into feature attributions.')
p_xai.paragraph_format.line_spacing = 1.15

add_styled_heading(doc, 'b) SHAP (SHapley Additive exPlanations)', 2)
p_shap = doc.add_paragraph('Proposed by Lundberg and Lee (2017), SHAP is grounded in Cooperative Game Theory, specifically Shapley values formulated by Lloyd Shapley in 1953. In SHAP, the model prediction is viewed as the total payout of a game, and each input feature is treated as a player collaborating in a grand coalition. The Shapley value computes the weighted average marginal contribution of a feature across all possible 2^M feature subsets. SHAP uniquely satisfies four fundamental mathematical axioms: (1) Efficiency (sum of feature attributions equals model output minus base value), (2) Symmetry (identical features receive identical values), (3) Dummy/Null player (features with zero marginal impact receive zero attribution), and (4) Additivity/Monotonicity. For decision trees and Random Forests, TreeSHAP optimizes the computation from exponential O(2^M) to polynomial O(TLD^2) where T is trees, L is leaves, and D is maximum depth.')
p_shap.paragraph_format.line_spacing = 1.15

add_styled_heading(doc, 'c) LIME (Local Interpretable Model-agnostic Explanations)', 2)
p_lime = doc.add_paragraph('Proposed by Ribeiro, Singh, and Guestrin (2016), LIME operates on the premise that while an entire global decision boundary may be highly non-linear and complex, the boundary immediately surrounding a single specific query point can be faithfully approximated by a simple, interpretable linear surrogate model. LIME draws synthetic perturbations uniformly around the query instance, weights each perturbation according to an exponential Euclidean distance kernel, queries the original black-box model to obtain predictions, and solves a locally weighted sparse linear regression. For tabular datasets, continuous variables are discretized into quartiles or deciles, producing easily readable condition rules such as worst radius > 18.79.')
p_lime.paragraph_format.line_spacing = 1.15

add_styled_heading(doc, 'd) Benefits and Limitations of SHAP and LIME', 2)
p_bl = doc.add_paragraph('SHAP provides rigorous theoretical guarantees, guarantees consistency across model architectures, and enables global dataset aggregation (mean absolute attributions) as well as local instance explanation. However, KernelSHAP can be computationally expensive on large feature spaces. LIME is completely model-agnostic and yields intuitive threshold rules, but its stochastic perturbation mechanism introduces minor variance between runs, and it lacks formal axiomatic guarantees.')
p_bl.paragraph_format.line_spacing = 1.15

# Architecture / Workflow Diagram
add_styled_heading(doc, 'Architecture / Workflow Diagram', 1)
p_diag_desc = doc.add_paragraph('The end-to-end XAI workflow consists of data ingestion, model fitting, metric evaluation, parallel explanation generation via TreeSHAP and LimeTabularExplainer, and interactive presentation through Streamlit.')
if os.path.exists('ml-explainability-shap-lime/outputs/architecture_diagram.png'):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_img = p_img.add_run()
    run_img.add_picture('ml-explainability-shap-lime/outputs/architecture_diagram.png', width=Inches(6.0))
p_fig1 = doc.add_paragraph('Figure 1: Machine Learning Model Training and Dual Explainability Architecture (SHAP & LIME)')
p_fig1.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_fig1.runs[0].font.italic = True

doc.add_page_break()

# Procedure (Step-by-Step Implementation)
add_styled_heading(doc, 'Procedure (Step-by-Step Implementation)', 1)

add_styled_heading(doc, 'Step 1: Dataset Inspection & Stratified Splitting (src/data_loader.py)', 2)
p_step1 = doc.add_paragraph('The Breast Cancer Wisconsin dataset is loaded directly via Scikit-Learn. We inspect the feature dimensionality (30 features, 569 instances), verify 0 missing values and 0 duplicate rows, and map class labels (0: Malignant, 1: Benign). The dataset is split into 80% training (455 samples) and 20% testing (114 samples) using stratified sampling (random_state=42) while retaining Pandas DataFrame column identities so feature names remain intact throughout all explanation pipelines.')

add_styled_heading(doc, 'Step 2: Model Training & Hyperparameter Setup (src/train.py)', 2)
p_step2 = doc.add_paragraph('A baseline Random Forest Classifier is trained using n_estimators=100 and random_state=42. The fitted model artifact is serialized to models/random_forest_model.joblib, and train/test splits are saved to CSV files in data/ for reproducible cross-module analysis.')

add_styled_heading(doc, 'Step 3: Comprehensive Evaluation on Test Set (src/evaluate.py)', 2)
p_step3 = doc.add_paragraph('The model is evaluated on the 114 unseen test samples. It achieves an accuracy of 95.61%, precision of 95.89%, recall of 97.22%, F1-score of 96.55%, and an ROC-AUC of 99.37%. Evaluation metrics are saved to JSON format, accompanied by high-resolution Confusion Matrix and ROC curve plots.')
if os.path.exists('ml-explainability-shap-lime/outputs/evaluation/confusion_matrix.png'):
    p_cm = doc.add_paragraph()
    p_cm.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cm.add_run().add_picture('ml-explainability-shap-lime/outputs/evaluation/confusion_matrix.png', width=Inches(3.8))
    p_cap_cm = doc.add_paragraph('Figure 2: Confusion Matrix for Test Set Predictions (Random Forest)')
    p_cap_cm.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap_cm.runs[0].font.italic = True

add_styled_heading(doc, 'Step 4: SHAP Global and Local Explainability (src/shap_explainer.py)', 2)
p_step4 = doc.add_paragraph('We initialize shap.TreeExplainer with the trained Random Forest model. To properly handle binary classification without dimensionality mismatch, we slice the 3D explanation tensor (N, 30, 2) along class index 1 (Benign). We generate global feature importance bar plots, beeswarm summary plots, and local waterfall plots for individual patients.')
if os.path.exists('ml-explainability-shap-lime/outputs/shap/shap_beeswarm_plot.png'):
    p_bs = doc.add_paragraph()
    p_bs.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_bs.add_run().add_picture('ml-explainability-shap-lime/outputs/shap/shap_beeswarm_plot.png', width=Inches(4.5))
    p_cap_bs = doc.add_paragraph('Figure 3: SHAP Beeswarm Summary Plot showing global feature value impacts')
    p_cap_bs.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap_bs.runs[0].font.italic = True

add_styled_heading(doc, 'Step 5: LIME Local Perturbation Explanations (src/lime_explainer.py)', 2)
p_step5 = doc.add_paragraph('We configure LimeTabularExplainer using the training data statistics and feature names. Continuous features are discretized into quartile rules. Explanations are computed for multiple test instances. For Sample 0 (Benign), rules like worst perimeter <= 84.11 provide strong positive weights (+0.16) toward Benign.')
if os.path.exists('ml-explainability-shap-lime/outputs/lime/lime_sample_0.png'):
    p_lime_img = doc.add_paragraph()
    p_lime_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_lime_img.add_run().add_picture('ml-explainability-shap-lime/outputs/lime/lime_sample_0.png', width=Inches(4.5))
    p_cap_lime = doc.add_paragraph('Figure 4: LIME Local Explanation Bar Chart for Test Sample 0')
    p_cap_lime.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap_lime.runs[0].font.italic = True

add_styled_heading(doc, 'Step 6: Comparative Analysis (src/compare_explainers.py)', 2)
p_step6 = doc.add_paragraph('A direct empirical comparison is conducted by plotting SHAP and LIME attributions side-by-side for identical test samples and documenting methodological differences in a comparison matrix CSV.')
if os.path.exists('ml-explainability-shap-lime/outputs/comparison/shap_vs_lime_side_by_side_sample_0.png'):
    p_comp_img = doc.add_paragraph()
    p_comp_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_comp_img.add_run().add_picture('ml-explainability-shap-lime/outputs/comparison/shap_vs_lime_side_by_side_sample_0.png', width=Inches(5.0))
    p_cap_comp = doc.add_paragraph('Figure 5: Side-by-Side Comparison of SHAP and LIME on Test Sample 0')
    p_cap_comp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap_comp.runs[0].font.italic = True

add_styled_heading(doc, 'Step 7: Automated Verification Suite (tests/test_pipeline.py)', 2)
p_step7 = doc.add_paragraph('Automated unit and integration tests are executed using Pytest, validating dataset loading, stratified proportions, prediction probabilities, SHAP feature name retention, LIME weights, and invalid dimension exception handling.')
if os.path.exists('ml-explainability-shap-lime/outputs/terminal_pytest_output.png'):
    p_test_img = doc.add_paragraph()
    p_test_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_test_img.add_run().add_picture('ml-explainability-shap-lime/outputs/terminal_pytest_output.png', width=Inches(5.2))
    p_cap_test = doc.add_paragraph('Figure 6: Automated Pytest execution confirming 7/7 passing tests')
    p_cap_test.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap_test.runs[0].font.italic = True

doc.add_page_break()

# Code / Configuration Files
add_styled_heading(doc, 'Code / Configuration Files', 1)
doc.add_paragraph('Key excerpts from the modular codebase are shown below. Full code is maintained in the project repository.')

doc.add_paragraph('1. Data Ingestion & Stratified Split (src/data_loader.py):', style='List Bullet')
code1 = '''def get_stratified_split(test_size=0.20, random_state=42):
    raw_data = load_breast_cancer(as_frame=True)
    X, y = raw_data.data.copy(), raw_data.target.copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, raw_data'''
p_c1 = doc.add_paragraph(code1)
p_c1.runs[0].font.name = 'Consolas'
p_c1.runs[0].font.size = Pt(8.5)

doc.add_paragraph('2. SHAP Explanation Slicing for Binary Classification (src/shap_explainer.py):', style='List Bullet')
code2 = '''def extract_class_shap_values(explanation, target_class=1):
    if len(explanation.shape) == 3 and explanation.shape[2] == 2:
        return shap.Explanation(
            values=explanation.values[:, :, target_class],
            base_values=explanation.base_values[:, target_class],
            data=explanation.data,
            feature_names=explanation.feature_names
        )
    return explanation'''
p_c2 = doc.add_paragraph(code2)
p_c2.runs[0].font.name = 'Consolas'
p_c2.runs[0].font.size = Pt(8.5)

doc.add_paragraph('3. LIME Tabular Explainer Initialization (src/lime_explainer.py):', style='List Bullet')
code3 = '''explainer = LimeTabularExplainer(
    training_data=np.ascontiguousarray(X_train.values),
    feature_names=list(X_train.columns),
    class_names=['Malignant', 'Benign'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)'''
p_c3 = doc.add_paragraph(code3)
p_c3.runs[0].font.name = 'Consolas'
p_c3.runs[0].font.size = Pt(8.5)

p_git = doc.add_paragraph('GitHub Repository Link: https://github.com/[username]/ml-explainability-shap-lime')
p_git.runs[0].font.bold = True

# Output / Results
add_styled_heading(doc, 'Output / Results', 1)
doc.add_paragraph('Empirical performance and evaluation metrics obtained on the test dataset:')

res_data = [
    ('Model Test Accuracy', '95.61% (109 / 114 correct classifications)', 'High Generalization'),
    ('Precision (Benign)', '95.89% (70 true positives / 73 positive calls)', 'Low False Positive Rate'),
    ('Recall (Benign)', '97.22% (70 true positives / 72 actual benign)', 'High Clinical Sensitivity'),
    ('F1-Score (Benign)', '96.55%', 'Balanced Performance'),
    ('ROC-AUC Score', '99.37%', 'Exceptional Class Separability'),
    ('Pytest Test Suite', '7 Passed / 0 Failed in 7.49s', '100% Pipeline Verification'),
    ('Top Decisive Biomarker', 'worst concave points & worst perimeter', 'Agreement between SHAP & LIME')
]
res_table = doc.add_table(rows=len(res_data)+1, cols=3)
res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
for j, text in enumerate(['Metric / Parameter', 'Observed Value', 'Status / Clinical Relevance']):
    c = res_table.rows[0].cells[j]
    c.text = text
    set_cell_background(c, '003366')
    for r in c.paragraphs[0].runs:
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

for i, (m, v, s) in enumerate(res_data):
    row = res_table.rows[i+1]
    row.cells[0].text = m
    row.cells[1].text = v
    row.cells[2].text = s
    set_cell_background(row.cells[0], 'F2F4F7')
    set_cell_background(row.cells[1], 'FFFFFF' if i % 2 == 0 else 'FAFAFA')
    set_cell_background(row.cells[2], 'FFFFFF' if i % 2 == 0 else 'FAFAFA')

doc.add_paragraph('')

# Observations and Analysis
add_styled_heading(doc, 'Observations and Analysis', 1)
obs = [
    '1. Cross-Method Agreement on Key Features: Both SHAP and LIME independently identified worst concave points, worst perimeter, and worst radius as the primary determinants of malignancy. Elevated values consistently reduced the probability of a Benign diagnosis.',
    '2. Global vs Local Perspectives: While LIME successfully generated intuitive decision rules for individual biopsy specimens (e.g., worst perimeter <= 84.11), SHAP beeswarm plots exposed dataset-wide non-linear patterns, showing long tails of severe negative impacts for extreme concave point measurements.',
    '3. Handling Binary Output Dimensions in Modern SHAP: In modern SHAP versions (>=0.45.0), TreeExplainer outputs an Explanation object with shape (N, num_features, 2). Slicing explicitly for class index 1 avoided dimensional broadcasting bugs while generating waterfall plots.',
    '4. LIME Discretization Utility: LIME discretization of continuous features into human-readable range thresholds (e.g., worst area <= 515.30) provides clinicians with immediate, intuitive heuristics compared to raw continuous regression coefficients.',
    '5. Computational Efficiency: TreeSHAP took under 0.5 seconds to compute exact Shapley attributions across all 114 test samples, whereas LIME required approximately 1.5 seconds per sample due to generating 5000 synthetic perturbations.'
]
for o in obs:
    p = doc.add_paragraph(o)
    p.paragraph_format.space_after = Pt(4)

# Conclusion
add_styled_heading(doc, 'Conclusion', 1)
p_conc = doc.add_paragraph('Thus, we have successfully built, evaluated, and demonstrated a comprehensive Machine Learning Explainability pipeline using Scikit-Learn, SHAP, and LIME on the Breast Cancer Wisconsin dataset. By integrating both cooperative game theory and local surrogate modeling, the project transforms black-box Random Forest predictions into transparent, clinically interpretable insights. The interactive Streamlit dashboard and automated test suite ensure the system is reproducible, verifiable, and well-suited for academic viva presentation.')
p_conc.paragraph_format.line_spacing = 1.15

# References
add_styled_heading(doc, 'References', 1)
refs = [
    '[1] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," Advances in Neural Information Processing Systems (NeurIPS 30), pp. 4765-4774, 2017.',
    '[2] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why Should I Trust You?: Explaining the Predictions of Any Classifier," Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 1135-1144, 2016.',
    '[3] C. Molnar, "Interpretable Machine Learning: A Guide for Making Black Box Models Explainable," 2nd ed., Leanpub, 2022. Available: https://christophm.github.io/interpretable-ml-book/',
    '[4] Scikit-Learn Developers, "Breast Cancer Wisconsin (Diagnostic) Dataset," Scikit-Learn Documentation, https://scikit-learn.org/stable/datasets/toy_dataset.html, accessed Nov 2026.'
]
for r in refs:
    p = doc.add_paragraph(r)
    p.paragraph_format.space_after = Pt(2)

doc.save('MLOps_Assignment_9_SHAP_LIME_Explainability_Report.docx')
print('Document generated successfully: MLOps_Assignment_9_SHAP_LIME_Explainability_Report.docx')
