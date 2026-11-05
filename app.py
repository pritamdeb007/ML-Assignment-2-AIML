"""PhishLens: Streamlit interface for evaluating saved phishing classifiers.

Run locally or in BITS Virtual Lab with:
    streamlit run app.py

This application does not train or retrain models. It loads the five saved
pipelines produced by training.py and evaluates them on a labelled test CSV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# Keep page configuration as the first Streamlit command.
st.set_page_config(
    page_title="PhishLens | Website Threat Analysis",
    page_icon="🛡️",
    layout="wide",
)

# Keep paths relative to this app.py file so deployment works on Streamlit Cloud.
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET_COLUMN = "Result"
PHISHING_LABEL = "Phishing"
LEGITIMATE_LABEL = "Legitimate"
LABEL_ORDER = [PHISHING_LABEL, LEGITIMATE_LABEL]

MODEL_FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# ---------------------------------------------------------------------------
# 1. Load metadata and saved models (no fitting or retraining takes place here)
# ---------------------------------------------------------------------------

@st.cache_data
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "model/metadata.json is missing. Check that the model folder is in the repository."
        )
    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


@st.cache_resource
def load_saved_models() -> Dict[str, object]:
    """Load the saved model pipelines once and reuse them between app reruns."""
    loaded_models = {}
    for model_name, file_name in MODEL_FILE_NAMES.items():
        model_path = MODEL_DIR / file_name
        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found: {model_path.relative_to(PROJECT_DIR)}. "
                "Check that all five model files are present in the model folder."
            )
        loaded_models[model_name] = joblib.load(model_path)
    return loaded_models


# ---------------------------------------------------------------------------
# 2. Validate the uploaded CSV before asking a model to make predictions
# ---------------------------------------------------------------------------

def convert_target_labels(target_values: pd.Series) -> pd.Series:
    """Accept either readable class names or the original -1/+1 target values."""
    mapping = {
        "-1": PHISHING_LABEL,
        "-1.0": PHISHING_LABEL,
        "phishing": PHISHING_LABEL,
        "+1": LEGITIMATE_LABEL,
        "1": LEGITIMATE_LABEL,
        "1.0": LEGITIMATE_LABEL,
        "legitimate": LEGITIMATE_LABEL,
    }
    converted = target_values.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        unknown_values = target_values[converted.isna()].astype(str).unique().tolist()
        raise ValueError(
            "The Result column contains unrecognised labels: "
            f"{unknown_values}. Use -1/+1 or Phishing/Legitimate."
        )
    return converted


def validate_uploaded_data(
    uploaded_df: pd.DataFrame,
    feature_names: list[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Check required columns, numeric features, missing values and target labels."""
    required_columns = feature_names + [TARGET_COLUMN]
    missing_columns = [name for name in required_columns if name not in uploaded_df.columns]
    if missing_columns:
        raise ValueError(
            "The uploaded CSV is missing these required columns: "
            + ", ".join(missing_columns)
        )

    if uploaded_df.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

    # Preserve the exact feature order recorded when the models were trained.
    X_uploaded = uploaded_df[feature_names].copy()
    for column in feature_names:
        X_uploaded[column] = pd.to_numeric(X_uploaded[column], errors="coerce")

    if X_uploaded.isna().any().any():
        bad_columns = X_uploaded.columns[X_uploaded.isna().any()].tolist()
        raise ValueError(
            "Some feature values are missing or non-numeric in these columns: "
            + ", ".join(bad_columns)
        )

    y_actual = convert_target_labels(uploaded_df[TARGET_COLUMN])
    return X_uploaded, y_actual


# ---------------------------------------------------------------------------
# 3. Evaluation helper: calculate exactly the six required evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_one_model(
    model: object,
    X_uploaded: pd.DataFrame,
    y_actual: pd.Series,
) -> Tuple[dict, pd.Series, dict, object]:
    """Predict with an already trained model and calculate evaluation results."""
    # predict() uses the fitted model; there is deliberately no model.fit() here.
    y_predicted = pd.Series(model.predict(X_uploaded), index=y_actual.index)

    class_names = list(model.classes_)
    phishing_column = class_names.index(PHISHING_LABEL)
    phishing_probabilities = model.predict_proba(X_uploaded)[:, phishing_column]
    actual_phishing = (y_actual == PHISHING_LABEL).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_actual, y_predicted)),
        "AUC": float(roc_auc_score(actual_phishing, phishing_probabilities)),
        "Precision": float(
            precision_score(
                y_actual,
                y_predicted,
                pos_label=PHISHING_LABEL,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_actual,
                y_predicted,
                pos_label=PHISHING_LABEL,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                y_actual,
                y_predicted,
                pos_label=PHISHING_LABEL,
                zero_division=0,
            )
        ),
        "MCC": float(matthews_corrcoef(y_actual, y_predicted)),
    }

    report = classification_report(
        y_actual,
        y_predicted,
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_actual, y_predicted, labels=LABEL_ORDER)
    return metrics, y_predicted, report, matrix


def format_metric(metric_name: str, metric_value: float) -> str:
    """Use percentage formatting for rate metrics and decimals for score metrics."""
    if metric_name in {"Accuracy", "Precision", "Recall"}:
        return f"{metric_value:.2%}"
    return f"{metric_value:.4f}"


# ---------------------------------------------------------------------------
# 4. PhishLens visual identity and page styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --phish-ink: #17283d;
            --phish-navy: #14263b;
            --phish-teal: #20c9b2;
            --phish-blue: #2f6df6;
            --phish-muted: #64748b;
            --phish-canvas: #f2f6fa;
            --phish-line: #dce5ee;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--phish-canvas);
        }
        [data-testid="stHeader"] {
            background: rgba(242, 246, 250, 0.92);
        }
        .block-container {
            max-width: 1120px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        .phish-hero {
            color: #f8fafc;
            background: linear-gradient(125deg, #13243a 0%, #1b354d 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 23px 27px 20px;
            margin: 0 0 22px 0;
            box-shadow: 0 12px 28px rgba(20, 38, 59, 0.12);
        }
        .phish-brand-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 17px;
        }
        .phish-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.16rem;
            font-weight: 750;
            letter-spacing: -0.02em;
        }
        .phish-brand svg {
            width: 25px;
            height: 25px;
            flex: 0 0 auto;
        }
        .phish-lab-tag {
            color: #5af0c6;
            background: rgba(32, 201, 178, 0.13);
            border: 1px solid rgba(90, 240, 198, 0.22);
            border-radius: 999px;
            padding: 5px 11px;
            font-size: 0.70rem;
            font-weight: 750;
            letter-spacing: 0.09em;
            white-space: nowrap;
        }
        .phish-hero h1 {
            color: #f8fafc;
            font-size: clamp(1.65rem, 3vw, 2.15rem);
            font-weight: 780;
            line-height: 1.15;
            letter-spacing: -0.035em;
            margin: 0 0 7px 0;
        }
        .phish-hero .phish-subtitle {
            color: #c5d2df;
            font-size: 0.88rem;
            margin: 0;
        }
        .phish-hero-divider {
            height: 1px;
            background: rgba(226, 232, 240, 0.22);
            margin: 18px 0 17px;
        }
        .phish-stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
        }
        .phish-stat-value {
            display: block;
            color: #ffffff;
            font-size: 1.34rem;
            font-weight: 760;
            line-height: 1.25;
            font-variant-numeric: tabular-nums;
        }
        .phish-stat-label {
            display: block;
            color: #c5d2df;
            font-size: 0.80rem;
            margin-top: 2px;
        }

        .section-heading {
            color: var(--phish-ink);
            font-size: 1.16rem;
            font-weight: 750;
            letter-spacing: -0.015em;
            margin: 0.15rem 0 0.2rem 0;
        }
        .section-hint {
            color: var(--phish-muted);
            font-size: 0.84rem;
            margin: 0 0 0.75rem 0;
        }
        .field-heading {
            color: var(--phish-ink);
            font-weight: 700;
            font-size: 0.91rem;
            margin: 0.1rem 0 0.45rem 0;
        }
        .workspace-note {
            color: var(--phish-muted);
            font-size: 0.82rem;
            margin: 0.1rem 0 0.8rem 0;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e3eaf1;
            border-radius: 13px;
            padding: 13px 15px 12px;
            min-height: 88px;
            box-shadow: 0 3px 10px rgba(23, 40, 61, 0.035);
        }
        [data-testid="stMetricLabel"] {
            color: #64748b;
            font-size: 0.78rem;
        }
        [data-testid="stMetricValue"] {
            color: #17283d;
            font-size: 1.42rem;
            font-weight: 740;
            font-variant-numeric: tabular-nums;
        }

        [data-testid="stFileUploader"] section {
            background: #fbfdff;
            border: 1px dashed #9bb2cf;
            border-radius: 12px;
        }
        [data-testid="stFileUploader"] section:hover {
            border-color: var(--phish-blue);
            background: #f7faff;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: #edf3f8;
            border-color: #dce5ee;
            border-radius: 10px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #e3eaf1;
            border-radius: 11px;
            overflow: hidden;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 7px;
            border-bottom: 1px solid var(--phish-line);
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            border-radius: 9px 9px 0 0;
            padding: 0 13px;
            color: #52647a;
            font-weight: 650;
        }
        .stTabs [aria-selected="true"] {
            color: #087f75;
            background: #e3f6f2;
        }
        [data-testid="stAlert"] {
            border-radius: 11px;
        }
        @media (max-width: 640px) {
            .phish-hero { padding: 19px 18px 17px; }
            .phish-lab-tag { font-size: 0.61rem; padding: 4px 8px; }
            .phish-stat-value { font-size: 1.12rem; }
            .phish-stat-label { font-size: 0.70rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 5. Load the project files and render the PhishLens workspace
# ---------------------------------------------------------------------------

try:
    project_metadata = load_metadata()
    saved_models = load_saved_models()
except Exception as error:
    st.error(str(error))
    st.stop()

feature_names = project_metadata["feature_names"]

# Warn only for a Scikit-learn version mismatch; do not alter or retrain models.
training_sklearn_version = (
    project_metadata.get("software_versions", {}).get("scikit_learn")
)
if training_sklearn_version and training_sklearn_version != sklearn.__version__:
    st.warning(
        "The saved models were created with Scikit-learn "
        f"{training_sklearn_version}, but this app is running "
        f"{sklearn.__version__}. Check compatibility if model loading or "
        "predictions behave unexpectedly."
    )

st.markdown(
    """
    <section class="phish-hero">
        <div class="phish-brand-row">
            <div class="phish-brand">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M12 2.6 20 5.8v5.7c0 5.1-3.3 8.5-8 10-4.7-1.5-8-4.9-8-10V5.8l8-3.2Z"
                          stroke="#36e2c1" stroke-width="1.9" stroke-linejoin="round"/>
                    <path d="m8.7 11.9 2.1 2.1 4.6-4.8" stroke="#36e2c1"
                          stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>PhishLens</span>
            </div>
            <span class="phish-lab-tag">CLASSIFICATION LAB</span>
        </div>
        <h1>Website Threat Analysis</h1>
        <p class="phish-subtitle">Machine learning research workspace · UCI Phishing Websites</p>
        <div class="phish-hero-divider"></div>
        <div class="phish-stats">
            <div><span class="phish-stat-value">11,055</span>
                 <span class="phish-stat-label">Dataset instances</span></div>
            <div><span class="phish-stat-value">30</span>
                 <span class="phish-stat-label">Input features</span></div>
            <div><span class="phish-stat-value">5</span>
                 <span class="phish-stat-label">Saved classifiers</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Experiment workspace</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-hint">Evaluate a labelled hold-out dataset with the five saved classifiers.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="field-heading">Test dataset</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload test CSV",
    type=["csv"],
    help=(
        "Upload the held-out test_data.csv generated by training.py. "
        "It must contain the 30 feature columns and the Result target column."
    ),
)

st.markdown('<div class="field-heading">Classification model</div>', unsafe_allow_html=True)
selected_model_name = st.selectbox(
    "Choose a saved classifier",
    options=list(MODEL_FILE_NAMES.keys()),
    label_visibility="collapsed",
)

st.markdown(
    '<div class="workspace-note">PhishLens loads saved model pipelines only. Uploading a CSV never triggers model training.</div>',
    unsafe_allow_html=True,
)

if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        X_uploaded, y_actual = validate_uploaded_data(uploaded_data, feature_names)
    except Exception as error:
        st.error(f"Unable to use this CSV: {error}")
        st.stop()

    st.success(
        f"Test dataset validated · {len(X_uploaded):,} rows · "
        f"{len(feature_names)} required input features."
    )

    with st.expander("Preview uploaded test data"):
        st.dataframe(uploaded_data.head(10), use_container_width=True)

    # Evaluate all five saved models on the same uploaded test data.
    all_results = {}
    all_reports = {}
    all_matrices = {}

    try:
        with st.spinner("Evaluating the five saved classifiers…"):
            for model_name, saved_model in saved_models.items():
                metrics, _predictions, report, matrix = evaluate_one_model(
                    saved_model, X_uploaded, y_actual
                )
                all_results[model_name] = metrics
                all_reports[model_name] = report
                all_matrices[model_name] = matrix
    except Exception as error:
        st.error(f"Model evaluation failed: {error}")
        st.stop()

    st.markdown('<div class="section-heading">Performance indicators</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-hint">Selected classifier: <b>{selected_model_name}</b></div>',
        unsafe_allow_html=True,
    )

    selected_metrics = all_results[selected_model_name]
    metric_rows = [
        [("Accuracy", "Accuracy"), ("AUC", "AUC"), ("Precision", "Precision")],
        [("Recall", "Recall"), ("F1", "F1 Score"), ("MCC", "MCC")],
    ]
    for metric_row in metric_rows:
        metric_columns = st.columns(3)
        for column, (metric_key, display_label) in zip(metric_columns, metric_row):
            column.metric(
                display_label,
                format_metric(metric_key, selected_metrics[metric_key]),
            )

    st.markdown('<div class="section-heading">Classification analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-hint">Inspect the selected model or compare all five classifiers on the same test data.</div>',
        unsafe_allow_html=True,
    )

    confusion_tab, comparison_tab, report_tab = st.tabs(
        ["Confusion matrix", "Model comparison", "Classification report"]
    )

    with confusion_tab:
        st.caption("Rows show actual classes; columns show predicted classes.")
        left_space, matrix_column, right_space = st.columns([1, 1.15, 1])
        with matrix_column:
            # Compact dimensions keep the chart readable without dominating the page.
            figure, axis = plt.subplots(figsize=(3.35, 2.85), dpi=115)
            display = ConfusionMatrixDisplay(
                confusion_matrix=all_matrices[selected_model_name],
                display_labels=LABEL_ORDER,
            )
            display.plot(
                ax=axis,
                values_format="d",
                colorbar=False,
                cmap="Blues",
            )
            axis.set_title(f"{selected_model_name}", fontsize=9, pad=8, weight="bold")
            axis.set_xlabel("Predicted class", fontsize=8, labelpad=6)
            axis.set_ylabel("Actual class", fontsize=8, labelpad=6)
            axis.tick_params(axis="both", labelsize=7)
            if display.text_ is not None:
                for cell_text in display.text_.ravel():
                    cell_text.set_fontsize(8)
                    cell_text.set_fontweight("semibold")
            figure.tight_layout(pad=1.0)
            st.pyplot(figure, clear_figure=True, use_container_width=False)
            plt.close(figure)

    with comparison_tab:
        comparison_table = pd.DataFrame.from_dict(all_results, orient="index")
        comparison_table.index.name = "ML Model Name"
        comparison_table = comparison_table.rename(columns={"F1": "F1 Score"})
        st.dataframe(comparison_table.round(4), use_container_width=True)
        st.caption(
            "All five classifiers are evaluated against the same uploaded test set. "
            "AUC uses the phishing class probability."
        )

    with report_tab:
        report_frame = pd.DataFrame(all_reports[selected_model_name]).transpose()
        st.dataframe(report_frame.round(4), use_container_width=True)
        st.caption("The positive class for Precision, Recall and F1 is Phishing.")

    st.info(
        "Evaluation uses the previously saved model pipelines. No model fitting "
        "or retraining is performed in this application."
    )
else:
    st.info("Upload the labelled test_data.csv file to view performance indicators and classification analysis.")

st.divider()
st.caption(
    "PhishLens · Dataset: UCI Machine Learning Repository — Phishing Websites (ID 327). "
    "Positive class for Precision, Recall, F1 and AUC: Phishing."
)
