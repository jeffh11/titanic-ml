"""
Titanic Survival Prediction — Random Forest Classifier
Demonstrates: data cleaning, feature engineering, model training,
cross-validation, hyperparameter tuning, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)


# ── 1. Load Data ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load Titanic dataset.
    Tries a public URL first; falls back to the local CSV in data/.
    To use the real Titanic data, download it from:
    https://www.kaggle.com/competitions/titanic/data
    and place titanic.csv in the data/ folder.
    """
    import urllib.request
    url = (
        "https://raw.githubusercontent.com/datasciencedojo/datasets"
        "/master/titanic.csv"
    )
    local_path = os.path.join(os.path.dirname(__file__), "..", "data", "titanic.csv")
    try:
        df = pd.read_csv(url)
        print(f"Loaded from URL: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception:
        df = pd.read_csv(local_path)
        print(f"Loaded from local file: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── 2. Exploratory Data Analysis ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """Plot survival rates by key features and save figures."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Survival Rate by Feature", fontsize=14, fontweight="bold")

    for ax, col, title in zip(
        axes,
        ["Pclass", "Sex", "Embarked"],
        ["Passenger Class", "Sex", "Port of Embarkation"],
    ):
        survival = df.groupby(col)["Survived"].mean().reset_index()
        sns.barplot(data=survival, x=col, y="Survived", ax=ax, palette="Blues_d")
        ax.set_title(title)
        ax.set_ylabel("Survival Rate")
        ax.set_ylim(0, 1)
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center", fontsize=9,
            )

    plt.tight_layout()
    plt.savefig("outputs/eda_survival_rates.png", dpi=150)
    plt.close()
    print("Saved: outputs/eda_survival_rates.png")


# ── 3. Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features:
    - Impute missing values
    - Extract title from Name
    - Create FamilySize and IsAlone features
    - Bin Age and Fare
    - Drop low-signal columns
    """
    df = df.copy()

    # Impute
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Extract title from name
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    rare_titles = df["Title"].value_counts()
    rare_titles = rare_titles[rare_titles < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mme": "Mrs", "Ms": "Miss", "Mlle": "Miss"})

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Bin Age and Fare
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, 100], labels=False)
    df["FareBin"] = pd.qcut(df["Fare"], q=4, labels=False, duplicates="drop")

    # Encode categoricals
    le = LabelEncoder()
    for col in ["Sex", "Embarked", "Title"]:
        df[col] = le.fit_transform(df[col].astype(str))

    # Drop unused columns
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    return df


# ── 4. Train & Evaluate ───────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame) -> None:
    """Train a Random Forest with GridSearchCV, evaluate, and plot results."""

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Hyperparameter search
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8, None],
        "min_samples_split": [2, 5],
    }
    base_rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(base_rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    print(f"\nBest params: {grid_search.best_params_}")

    # Cross-validation
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Test set evaluation
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nTest ROC-AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

    # Confusion matrix
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=["Did not survive", "Survived"],
    ).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    # Feature importances
    importances = pd.Series(best_rf.feature_importances_, index=X.columns)
    importances.sort_values().plot(kind="barh", ax=axes[2], color="steelblue")
    axes[2].set_title("Feature Importances")
    axes[2].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig("outputs/model_evaluation.png", dpi=150)
    plt.close()
    print("\nSaved: outputs/model_evaluation.png")


# ── 5. Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_raw = load_data()
    run_eda(df_raw)
    df_clean = engineer_features(df_raw)
    train_and_evaluate(df_clean)
    print("\nDone.")