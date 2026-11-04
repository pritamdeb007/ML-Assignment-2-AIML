"""Streamlit application for evaluating the saved phishing classifiers.

Run locally or in BITS Virtual Lab with:
    streamlit run app.py

The app DOES NOT train models. It loads the five pipelines created by
training.py and evaluates them on an uploaded, labelled test CSV.
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
            "model/metadata.json is missing. Run python training.py first."
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
                "Run python training.py and upload the model folder to GitHub."
            )
        loaded_models[model_name] = joblib.load(model_path)
    return loaded_models


# ---------------------------------------------------------------------------
# 2. Validate the uploaded CSV before asking a model to make predictions
# ---------------------------------------------------------------------------

def convert_target_labels(target_values: pd.Series) -> pd.Series:
    """Accept either the readable class names or the original -1/+1 values."""
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

    # Use the feature order saved at training time. Extra columns, if any, are ignored.
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
# 3. Evaluation helper: calculate exactly the six BITS metrics
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


# ---------------------------------------------------------------------------
# 4. Streamlit page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Phishing Website Classifier",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Phishing Website Classification")
st.caption(
    "Machine Learning Assignment 2 | Compare five trained classifiers "
    "using a labelled test CSV."
)
st.markdown(
    "This application loads saved models. It does **not** train or retrain "
    "a model when a CSV is uploaded."
)

try:
    project_metadata = load_metadata()
    saved_models = load_saved_models()
except Exception as error:
    st.error(str(error))
    st.stop()

feature_names = project_metadata["feature_names"]

# Joblib model files can be sensitive to Scikit-learn version changes.
# Warn the evaluator if the app environment differs from the training environment.
training_sklearn_version = project_metadata.get("software_versions", {}).get("scikit_learn")
if training_sklearn_version and training_sklearn_version != sklearn.__version__:
    st.warning(
        "The saved models were created with Scikit-learn "
        f"{training_sklearn_version}, but this app is running "
        f"{sklearn.__version__}. Use the same version in both environments "
        "to avoid model-loading or prediction compatibility issues."
    )

st.subheader("1. Upload the labelled test dataset")
st.write(
    "Upload the `test_data.csv` generated by `training.py`. "
    "It must contain all 30 feature columns and the `Result` target column."
)
uploaded_file = st.file_uploader(
    "Choose a test CSV file",
    type=["csv"],
    help="Use the held-out test CSV. Do not upload training data or a new unlabelled dataset.",
)

if uploaded_file is not None:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        X_uploaded, y_actual = validate_uploaded_data(uploaded_data, feature_names)
    except Exception as error:
        st.error(f"Unable to use this CSV: {error}")
        st.stop()

    st.success(
        f"Dataset accepted: {len(X_uploaded):,} test rows and "
        f"{len(feature_names)} required input features."
    )

    with st.expander("Preview uploaded test data"):
        st.dataframe(uploaded_data.head(10), use_container_width=True)

    st.subheader("2. Select a classification model")
    selected_model_name = st.selectbox(
        "Choose one of the five saved models",
        options=list(MODEL_FILE_NAMES.keys()),
    )

    # Evaluate all five saved models so the evaluator can compare them on the
    # same uploaded test CSV, while also displaying details for the selected one.
    all_results = {}
    all_predictions = {}
    all_reports = {}
    all_matrices = {}

    try:
        for model_name, saved_model in saved_models.items():
            metrics, predictions, report, matrix = evaluate_one_model(
                saved_model, X_uploaded, y_actual
            )
            all_results[model_name] = metrics
            all_predictions[model_name] = predictions
            all_reports[model_name] = report
            all_matrices[model_name] = matrix
    except Exception as error:
        st.error(f"Model evaluation failed: {error}")
        st.stop()

    st.subheader("3. Evaluation metrics — selected model")
    selected_metrics = all_results[selected_model_name]
    metric_columns = st.columns(3)
    for column, metric_name in zip(
        metric_columns,
        ["Accuracy", "AUC", "Precision"],
    ):
        column.metric(metric_name, f"{selected_metrics[metric_name]:.4f}")

    metric_columns = st.columns(3)
    for column, metric_name in zip(
        metric_columns,
        ["Recall", "F1", "MCC"],
    ):
        column.metric(metric_name, f"{selected_metrics[metric_name]:.4f}")

    st.subheader("4. Confusion matrix")
    figure, axis = plt.subplots(figsize=(5.5, 4.2))
    display = ConfusionMatrixDisplay(
        confusion_matrix=all_matrices[selected_model_name],
        display_labels=LABEL_ORDER,
    )
    display.plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title(f"{selected_model_name} — Confusion Matrix")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)
    plt.close(figure)

    st.subheader("Classification report")
    report_frame = pd.DataFrame(all_reports[selected_model_name]).transpose()
    st.dataframe(report_frame.round(4), use_container_width=True)

    st.subheader("5. Comparison of all five models on this uploaded test CSV")
    comparison_table = pd.DataFrame.from_dict(all_results, orient="index")
    comparison_table.index.name = "ML Model Name"
    st.dataframe(comparison_table.round(4), use_container_width=True)

    st.info(
        "Evaluation is performed on the uploaded labelled test data using the "
        "previously saved models. No model fitting or retraining is performed here."
    )
else:
    st.info("Upload the `test_data.csv` file to begin evaluation.")

st.divider()
st.caption(
    "Dataset: UCI Machine Learning Repository — Phishing Websites (ID 327). "
    "Positive class for Precision, Recall, F1 and AUC: Phishing."
)
