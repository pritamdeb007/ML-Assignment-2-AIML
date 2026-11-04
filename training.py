"""
Train and evaluate five classification models using the
Phishing Websites dataset.

The program loads the dataset, checks the data, prepares
the training and testing sets, trains five classifiers,
calculates evaluation metrics and saves the trained
models and test dataset for the Streamlit application.

"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.io import arff
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# 1. Project settings
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_CANDIDATES = [
    PROJECT_DIR / "data" / "raw" / "phishing_websites.arff",
    PROJECT_DIR / "data" / "raw" / "Training Dataset.arff",
]
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
METRICS_PATH = PROJECT_DIR / "metrics_summary.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"

RANDOM_SEED = 42
N_SPLITS = 5  # First fold gives an approximately 80:20 train/test split.
TARGET_COLUMN = "Result"
PHISHING_LABEL = "Phishing"
LEGITIMATE_LABEL = "Legitimate"


# ---------------------------------------------------------------------------
# 2. Read and validate the original ARFF dataset
# ---------------------------------------------------------------------------

def find_dataset_file() -> Path:
    """Find the ARFF file without requiring the user to rename it."""
    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate

    expected = "\n".join(str(path) for path in RAW_DATA_CANDIDATES)
    raise FileNotFoundError(
        "The Phishing Websites ARFF file was not found. Place the downloaded "
        "file in data/raw. The script accepts either of these names:\n" + expected
    )


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load ARFF data and convert ARFF byte values into ordinary numbers.

    ARFF libraries can return values such as b'-1' rather than -1.
    This function decodes those values and converts every column to numeric.
    """
    raw_records, _metadata = arff.loadarff(str(file_path))
    data = pd.DataFrame(raw_records)

    for column in data.columns:
        if data[column].dtype == object:
            data[column] = data[column].map(
                lambda value: value.decode("utf-8")
                if isinstance(value, (bytes, bytearray))
                else value
            )
        data[column] = pd.to_numeric(data[column], errors="raise")

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Expected target column '{TARGET_COLUMN}' was not found. "
            f"Columns found: {list(data.columns)}"
        )

    if data.isna().any().any():
        missing_by_column = data.isna().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]
        raise ValueError(
            "Missing values were found. Review these columns before training:\n"
            + missing_by_column.to_string()
        )

    # This version of the UCI dataset encodes its two target classes as -1 and +1.
    observed_targets = set(data[TARGET_COLUMN].unique().tolist())
    if observed_targets != {-1, 1}:
        raise ValueError(
            "Unexpected target values. Expected -1 (Phishing) and +1 "
            f"(Legitimate), but found: {sorted(observed_targets)}"
        )

    return data


def print_dataset_inspection(data: pd.DataFrame) -> Dict[str, object]:
    """Print the dataset facts needed for the README and assignment write-up."""
    feature_columns = [column for column in data.columns if column != TARGET_COLUMN]

    # Exact duplicate rows are counted, but not automatically deleted.
    # The source file has no unique website ID. Identical feature rows can
    # represent repeated feature profiles, so deleting them without evidence
    # could change the source dataset's observations and class distribution.
    exact_duplicate_rows = int(data.duplicated().sum())
    repeated_feature_rows = int(data[feature_columns].duplicated().sum())

    # Count feature profiles that occur with more than one target label.
    # Such profiles are not distinguishable from the available 30 predictors.
    profile_label_counts = data.groupby(feature_columns, dropna=False)[TARGET_COLUMN].nunique()
    conflicting_profile_count = int((profile_label_counts > 1).sum())

    class_counts_raw = data[TARGET_COLUMN].value_counts().sort_index()
    class_counts = {
        PHISHING_LABEL: int(class_counts_raw.get(-1, 0)),
        LEGITIMATE_LABEL: int(class_counts_raw.get(1, 0)),
    }

    print("\n" + "=" * 72)
    print("DATASET INSPECTION — UCI PHISHING WEBSITES")
    print("=" * 72)
    print(f"Dataset rows (instances): {len(data):,}")
    print(f"Total columns including target: {data.shape[1]}")
    print(f"Input features (excluding target): {len(feature_columns)}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Missing values across the dataset: {int(data.isna().sum().sum())}")
    print(f"Exact repeated rows after the first occurrence: {exact_duplicate_rows:,}")
    print(f"Repeated feature profiles after the first occurrence: {repeated_feature_rows:,}")
    print(f"Feature profiles seen with both target labels: {conflicting_profile_count:,}")
    print("\nTarget distribution:")
    for label, count in class_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {label:12s}: {count:6,} ({percentage:5.2f}%)")

    print("\nInput feature names:")
    for number, name in enumerate(feature_columns, start=1):
        print(f"  {number:02d}. {name}")

    print("\nImportant data-handling decision:")
    print(
        "  Repeated rows are retained because the ARFF file does not contain "
        "a unique website identifier proving that they are erroneous records."
    )
    print(
        "  To reduce train/test leakage, identical 30-feature profiles are "
        "kept together in the same split group."
    )
    print("=" * 72 + "\n")

    return {
        "dataset_rows": int(len(data)),
        "total_columns_including_target": int(data.shape[1]),
        "number_of_features": int(len(feature_columns)),
        "target_column": TARGET_COLUMN,
        "missing_value_count": int(data.isna().sum().sum()),
        "exact_duplicate_rows_after_first": exact_duplicate_rows,
        "repeated_feature_profiles_after_first": repeated_feature_rows,
        "feature_profiles_with_conflicting_labels": conflicting_profile_count,
        "class_distribution_full": class_counts,
        "feature_names": feature_columns,
    }


# ---------------------------------------------------------------------------
# 3. Prepare the target and create a leakage-aware 80:20 split
# ---------------------------------------------------------------------------

def prepare_features_and_target(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate input features (X) from the answer column (y)."""
    X = data.drop(columns=[TARGET_COLUMN]).copy()

    # Make the target readable in the reports and Streamlit interface.
    # -1 means Phishing; +1 means Legitimate in this dataset version.
    y = data[TARGET_COLUMN].map(
        {-1: PHISHING_LABEL, 1: LEGITIMATE_LABEL}
    )

    if y.isna().any():
        raise ValueError("Some target labels could not be mapped to class names.")

    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Create an approximately 80:20 split while keeping matching profiles together.

    Why group identical feature profiles?
    This dataset contains many repeated combinations of its 30 coded features.
    If the same feature profile appears in both training and test sets, the
    test can be easier than testing on a genuinely unseen feature profile.
    StratifiedGroupKFold tries to preserve class proportions while keeping
    each feature-profile group wholly in either training or test data.
    """
    # A group is one unique combination of all 30 input feature values.
    # MultiIndex factorization assigns the same group number to matching rows.
    groups = pd.factorize(pd.MultiIndex.from_frame(X), sort=True)[0]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))

    # A simple safety check: no identical feature-profile group may cross splits.
    train_groups = set(groups[train_indices].tolist())
    test_groups = set(groups[test_indices].tolist())
    if train_groups.intersection(test_groups):
        raise RuntimeError("Feature-profile leakage detected across the split.")

    split_details = {
        "method": "StratifiedGroupKFold",
        "folds": N_SPLITS,
        "selected_fold": 1,
        "approximate_split_ratio": "80:20",
        "stratified_by": TARGET_COLUMN,
        "grouped_by": "identical combinations of all 30 input features",
        "random_seed": RANDOM_SEED,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "train_class_distribution": {
            str(key): int(value)
            for key, value in y.iloc[train_indices].value_counts().to_dict().items()
        },
        "test_class_distribution": {
            str(key): int(value)
            for key, value in y.iloc[test_indices].value_counts().to_dict().items()
        },
        "feature_profile_groups_in_train": int(len(train_groups)),
        "feature_profile_groups_in_test": int(len(test_groups)),
        "overlapping_feature_profile_groups": 0,
    }

    return train_indices, test_indices, split_details


# ---------------------------------------------------------------------------
# 4. Define the five required Scikit-learn classifiers
# ---------------------------------------------------------------------------

def build_models() -> Dict[str, ClassifierMixin]:
    """Build five models; scale features where distance/margins need it.

    StandardScaler is included inside the relevant Pipeline. This is important:
    the scaler learns its averages and standard deviations from training data
    only, and the saved pipeline repeats the same transformation in the app.
    """
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "K-Nearest Neighbors": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
                ),
            ]
        ),
    }


MODEL_FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


# ---------------------------------------------------------------------------
# 5. Calculate the six metrics required by BITS
# ---------------------------------------------------------------------------

def calculate_metrics(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Calculate all required metrics, treating Phishing as the positive class."""
    predictions = model.predict(X_test)

    # AUC needs a probability/score for the positive class, not only hard labels.
    if not hasattr(model, "predict_proba"):
        raise TypeError("This model must support predict_proba for the AUC calculation.")

    class_names = list(model.classes_)
    phishing_column = class_names.index(PHISHING_LABEL)
    phishing_probabilities = model.predict_proba(X_test)[:, phishing_column]
    actual_phishing = (y_test == PHISHING_LABEL).astype(int)

    return {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "AUC": float(roc_auc_score(actual_phishing, phishing_probabilities)),
        "Precision": float(
            precision_score(
                y_test,
                predictions,
                pos_label=PHISHING_LABEL,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_test,
                predictions,
                pos_label=PHISHING_LABEL,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                y_test,
                predictions,
                pos_label=PHISHING_LABEL,
                zero_division=0,
            )
        ),
        "MCC": float(matthews_corrcoef(y_test, predictions)),
    }


# ---------------------------------------------------------------------------
# 6. Main run: train, evaluate, and save all required outputs
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full training workflow from the original ARFF file."""
    dataset_path = find_dataset_file()
    print(f"Reading original dataset from: {dataset_path.relative_to(PROJECT_DIR)}")

    data = load_dataset(dataset_path)
    dataset_info = print_dataset_inspection(data)
    X, y = prepare_features_and_target(data)

    train_indices, test_indices, split_details = make_train_test_split(X, y)
    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    print("TRAIN/TEST SPLIT")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows : {len(X_test):,}")
    print("Training classes:", y_train.value_counts().to_dict())
    print("Testing classes :", y_test.value_counts().to_dict())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = build_models()
    metric_rows = []

    print("\nTRAINING AND EVALUATION RESULTS")
    print("-" * 100)

    for model_name, model in models.items():
        print(f"Training {model_name} ...")
        model.fit(X_train, y_train)

        metrics = calculate_metrics(model, X_test, y_test)
        metric_rows.append({"ML Model Name": model_name, **metrics})

        model_path = MODEL_DIR / MODEL_FILE_NAMES[model_name]
        joblib.dump(model, model_path)

        print(
            f"  Accuracy={metrics['Accuracy']:.4f} | "
            f"AUC={metrics['AUC']:.4f} | "
            f"Precision={metrics['Precision']:.4f} | "
            f"Recall={metrics['Recall']:.4f} | "
            f"F1={metrics['F1']:.4f} | "
            f"MCC={metrics['MCC']:.4f}"
        )
        print(f"  Saved trained pipeline: {model_path.relative_to(PROJECT_DIR)}")

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(METRICS_PATH, index=False, float_format="%.8f")

    # Save the held-out test rows with the actual target. The Streamlit app
    # needs the target column to calculate the evaluation metrics.
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.values
    test_data.to_csv(TEST_DATA_PATH, index=False)

    metadata = {
        "project_title": "Phishing Website Classification Using Machine Learning",
        "dataset_name": "Phishing Websites",
        "dataset_source": "UCI Machine Learning Repository, dataset ID 327",
        "dataset_url": "https://archive.ics.uci.edu/dataset/327/phishing",
        "target_column": TARGET_COLUMN,
        "positive_class_for_metrics": PHISHING_LABEL,
        "class_labels": [PHISHING_LABEL, LEGITIMATE_LABEL],
        "original_target_mapping": {"-1": PHISHING_LABEL, "1": LEGITIMATE_LABEL},
        "feature_names": dataset_info["feature_names"],
        "model_file_names": MODEL_FILE_NAMES,
        "dataset_inspection": dataset_info,
        "split_details": split_details,
        "metric_names": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
        "software_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print("\nMODEL COMPARISON TABLE")
    print(metrics_table.round(4).to_string(index=False))
    print("\nOUTPUT FILES CREATED")
    print(f"  Test data       : {TEST_DATA_PATH.relative_to(PROJECT_DIR)}")
    print(f"  Model folder    : {MODEL_DIR.relative_to(PROJECT_DIR)}")
    print(f"  Metrics summary : {METRICS_PATH.relative_to(PROJECT_DIR)}")
    print(f"  Model metadata  : {METADATA_PATH.relative_to(PROJECT_DIR)}")
    print("\nTraining completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise