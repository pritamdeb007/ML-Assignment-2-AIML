
# Machine Learning Assignment 2 AIMLZG565

## Project Title

**Phishing Website Classification Using Machine Learning**

## 1. Problem Statement

The aim of this project is to compare five supervised machine learning classification algorithms for identifying whether a website is phishing or legitimate based on its website-related characteristics.

The project uses the Phishing Websites dataset from the UCI Machine Learning Repository and evaluates the performance of five classification models using six evaluation metrics.

## 2. Dataset Description

| Parameter | Details |
|:--|:--|
| Dataset Name | Phishing Websites |
| Source | UCI Machine Learning Repository |
| Dataset ID | 327 |
| Problem Type | Binary Classification |
| Total Instances | 11,055 |
| Input Features | 30 |
| Target Variable | Result |
| Missing Values | 0 |
| Phishing Records | 4,898 |
| Legitimate Records | 6,157 |

**Dataset Source:** [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing)

**Target labels:**
- `-1`: Phishing
- `+1`: Legitimate

## 3. Data Preparation and Methodology

The dataset was inspected for missing values, duplicate records and target class distribution.

The original dataset contains 11,055 records and 30 input features, with no missing values.

The target variable, `Result`, was separated from the input features before model training. The original target values were mapped to the following class labels:

- `-1`: Phishing
- `+1`: Legitimate

A `StratifiedGroupKFold` approach was used with five folds, shuffling enabled and random seed 42. The first fold was selected, resulting in an approximately 80:20 training and testing split.

The resulting dataset was divided as follows:

| Dataset | Number of Records |
|:--|--:|
| Training dataset | 8,843 |
| Testing dataset | 2,212 |
| Total | 11,055 |

### Class Distribution

The class distribution across the full dataset and the training and testing partitions is shown below.

| Dataset Partition | Phishing (-1) | Legitimate (+1) | Total Records |
|:--|--:|--:|--:|
| Full dataset | 4,898 | 6,157 | 11,055 |
| Training dataset | 3,918 | 4,925 | 8,843 |
| Testing dataset | 980 | 1,232 | 2,212 |

Repeated feature profiles were retained rather than deleted. Identical combinations of the 30 input features were assigned to the same group to prevent the same feature profile from appearing in both training and testing datasets.

`StandardScaler` was included in pipelines for Logistic Regression, K-Nearest Neighbors and Support Vector Machine.

Phishing was treated as the positive class when calculating Precision, Recall, F1 Score and AUC.

## 4. Machine Learning Models

The following five supervised classification algorithms were implemented using Scikit-learn:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The trained models were saved using Joblib for subsequent use in the application.

## 5. Model Evaluation Metrics

Each model was evaluated using the following six metrics:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 6. Model Comparison

The following table presents the performance of all five classification models on the held-out test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.9317 | 0.9810 | 0.9377 | 0.9061 | 0.9216 | 0.8616 |
| Decision Tree | 0.9236 | 0.9324 | 0.9159 | 0.9112 | 0.9136 | 0.8451 |
| K-Nearest Neighbors | 0.9168 | 0.9640 | 0.9045 | 0.9082 | 0.9063 | 0.8315 |
| Gaussian Naive Bayes | 0.5995 | 0.9732 | 0.5252 | 1.0000 | 0.6887 | 0.3841 |
| Support Vector Machine | **0.9539** | **0.9870** | **0.9720** | 0.9224 | **0.9466** | **0.9070** |

## 7. Observations on Model Performance

The following observations were made based on the evaluation results:

- **Support Vector Machine (SVM):** Achieved the highest Accuracy of 0.9539 (95.39%), AUC of 0.9870 and MCC of 0.9070. It also achieved the highest Precision and F1 Score among the five models.
- **Logistic Regression:** Delivered strong performance with 93.17% Accuracy and an AUC of 0.9810.
- **Decision Tree:** Achieved 92.36% Accuracy and offers a relatively interpretable classification approach.
- **K-Nearest Neighbors:** Achieved 91.68% Accuracy with competitive Precision and Recall.
- **Gaussian Naive Bayes:** Achieved 100% Recall but comparatively low Precision of 0.5252. This indicates that while it identified all actual phishing cases in the test dataset, it also produced a relatively high number of false-positive predictions.

Overall, Support Vector Machine provided the strongest results across most of the evaluation metrics in this experiment. Gaussian Naive Bayes achieved the highest Recall, demonstrating why multiple evaluation metrics are important when comparing classification models.

## 8. GitHub Repository

The project source code, dataset, trained models and evaluation results are maintained in the following public GitHub repository:

[ML-Assignment-2-AIML](https://github.com/pritamdeb007/ML-Assignment-2-AIML)

## 9. Streamlit Application

The application source code is maintained in `app.py`.

The application is designed to provide the following functionality:

1. Upload a CSV test dataset.
2. Select one of the five trained classification models.
3. Display the six evaluation metrics.
4. Display a confusion matrix and classification report.

The application loads previously trained models and does not retrain them when a test dataset is uploaded.

**Deployment URL:** 

https://phishlens-aiml-pritam.streamlit.app/

## 10. Project Files

| File / Folder | Description |
|:--|:--|
| `training.py` | Loads and checks the ARFF dataset, prepares the data, trains and evaluates the five models, and saves the outputs. |
| `app.py` | Streamlit application for dataset upload and model evaluation. |
| `requirements.txt` | Python libraries and dependencies required by the project. |
| `test_data.csv` | Held-out test records containing 30 input features and the target column. |
| `metrics_summary.csv` | Evaluation results for all five classification models. |
| `model/` | Contains the five saved model pipelines and `metadata.json`. |
| `data/raw/` | Contains the original UCI ARFF dataset. |
| `FEATURE_REFERENCE.md` | Descriptions of the dataset's input features. |

## 11. Dataset Acknowledgement

The Phishing Websites dataset was obtained from the UCI Machine Learning Repository.

**Dataset:** Phishing Websites  
**Authors:** Mohammad, Rami M., Fadi Thabtah and Lee McCluskey.  
**Dataset ID:** 327

[UCI Machine Learning Repository – Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing)
