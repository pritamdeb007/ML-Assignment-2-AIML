# Machine Learning Assignment 2

## Project title
**Phishing Website Classification Using Machine Learning**

## Problem statement
The aim of this project is to compare five supervised classification algorithms for identifying whether a website is phishing or legitimate using the website-characteristic features in the UCI Phishing Websites dataset.

## Dataset Description

Dataset Name: Phishing Websites

Source: UCI Machine Learning Repository
Dataset ID: 327

Dataset URL:
https://archive.ics.uci.edu/dataset/327/phishing

Problem Type: Binary Classification

Total Instances: 11,055
Input Features: 30
Target Variable: Result

Target Labels:
- -1: Phishing
- +1: Legitimate

Missing Values: 0

## Data Preparation and Methodology

The dataset was inspected for missing values,
duplicate records and target class distribution.

The original dataset contains 11,055 records
and 30 input features.

The target variable was separated from the
input features before model training.

A Stratified Group K Fold approach was used
with 5 folds and random seed 42.

Identical feature profiles were kept within
the same partition to reduce data leakage.

The selected split contained:
- Training records: 8,843
- Testing records: 2,212

StandardScaler was included in pipelines
for Logistic Regression, KNN and SVM.

## Model Comparison

The following table presents the performance of the five
classification models using the six evaluation metrics.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.9317 | 0.9810 | 0.9377 | 0.9061 | 0.9216 | 0.8616 |
| Decision Tree | 0.9236 | 0.9324 | 0.9159 | 0.9112 | 0.9136 | 0.8451 |
| K-Nearest Neighbors | 0.9168 | 0.9640 | 0.9045 | 0.9082 | 0.9063 | 0.8315 |
| Gaussian Naive Bayes | 0.5995 | 0.9732 | 0.5252 | 1.0000 | 0.6887 | 0.3841 |
| Support Vector Machine | **0.9539** | **0.9870** | **0.9720** | 0.9224 | **0.9466** | **0.9070** |

## GitHub repository link
https://github.com/pritamdeb007/ML-Assignment-2-AIML

## Observations on model performance
Performance observations are noted as below:

- **Support Vector Machine (SVM):** Achieved the highest accuracy of 95.39%, AUC of 0.9870 and MCC of 0.9070 in this experiment.
- **Logistic Regression:** Delivered strong performance with 93.17% accuracy and 0.9810 AUC.
- **Decision Tree:** Achieved 92.36% accuracy and provided a relatively interpretable classification approach.
- **K-Nearest Neighbors:** Achieved 91.68% accuracy with competitive precision and recall.
- **Gaussian Naive Bayes:** Achieved 100% recall but comparatively low precision of 0.5252, indicating a higher number of false-positive predictions.

Overall, SVM provided the strongest balance of classification performance across the six evaluation metrics in this experiment. Gaussian Naive Bayes demonstrated the importance of considering precision and recall alongside accuracy.

## Streamlit Application

The application source code is maintained
in app.py.

The application is designed to provide:

1. CSV test dataset upload.
2. Selection among five trained classifiers.
3. Display of six evaluation metrics.
4. Confusion matrix or classification report.

The application uses previously saved
trained models and does not retrain them
when a test dataset is uploaded.

Deployment link: Pending.

## Project files

- `training.py` — loads and checks the ARFF dataset, creates the split, trains and evaluates the five models and saves the outputs.
- `app.py` — Streamlit interface for test-data upload and model evaluation.
- `requirements.txt` — Python dependencies.
- `test_data.csv` — held-out test rows, including the target column.
- `model/` — five saved model pipelines and `metadata.json`.
- `metrics_summary.csv` — six evaluation metrics for all five models.
- `data/raw/` — original UCI ARFF data file.
- `FEATURE_REFERENCE.md` — plain-language descriptions of the feature names.

## Dataset acknowledgement

Mohammad, Rami M., Fadi Thabtah, and Lee McCluskey. *Phishing Websites* dataset, UCI Machine Learning Repository, dataset ID 327. [Dataset record](https://archive.ics.uci.edu/dataset/327/phishing).
The Phishing Websites dataset was obtained
from the UCI Machine Learning Repository.
Dataset ID: 327
https://archive.ics.uci.edu/dataset/327/phishing
