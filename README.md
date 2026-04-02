# Titanic Survival Prediction

A machine learning pipeline that predicts passenger survival on the Titanic using a Random Forest classifier. Built to demonstrate core ML fundamentals: data cleaning, feature engineering, model training, hyperparameter tuning, and evaluation.

## What This Project Covers

| Step | Description |
|---|---|
| **EDA** | Survival rates broken down by class, sex, and embarkation port |
| **Feature Engineering** | Title extraction, family size, age/fare binning, missing value imputation |
| **Model** | Random Forest Classifier |
| **Tuning** | GridSearchCV over n_estimators, max_depth, min_samples_split |
| **Evaluation** | Cross-validation, ROC-AUC, confusion matrix, classification report |

## Usage

pip install -r requirements.txt
python src/model.py