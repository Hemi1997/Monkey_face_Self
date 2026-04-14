import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def run_classifier(
    X,
    y,
    model_type="svm",
    n_splits=5,
    random_state=42
):
    """
    Core classification pipeline:
    - imputation
    - standardization
    - classifier (default: SVM)
    - KFold cross-validation

    Returns:
    - dict with mean metrics
    """

    # -------------------------
    # Model selection
    # -------------------------
    if model_type == "svm":
        model = SVC(kernel="linear", probability=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # -------------------------
    # Pipeline
    # -------------------------
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # -------------------------
    # Cross-validation
    # -------------------------
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    acc_scores = []
    f1_scores = []
    auc_scores = []

    for train_idx, test_idx in kf.split(X):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # fit
        pipeline.fit(X_train, y_train)

        # predict
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # metrics
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        auc_scores.append(roc_auc_score(y_test, y_prob))

    return {
        "accuracy": np.mean(acc_scores),
        "f1": np.mean(f1_scores),
        "auc": np.mean(auc_scores)
    }



def select_feature_stats(X, stats="all"):
    """
    Select features based on stat suffix.

    stats:
    - "all"
    - list like ["mean", "max"]
    """

    if stats == "all":
        return X

    if isinstance(stats, str):
        stats = [stats]

    cols = [
        c for c in X.columns
        if any(c.endswith(f"__{s}") for s in stats)
    ]

    return X[cols]


def run_with_feature_selection(dataset, stats="all"):
    X = dataset.drop(columns=["label"], errors="ignore")
    y = dataset["label"]

    X = select_feature_stats(X, stats)

    return run_classifier(X, y)


def get_au_columns(dataset):
    au_map = {}

    for col in dataset.columns:
        if "__" not in col:
            continue

        au = col.split("__")[0]
        au_map.setdefault(au, []).append(col)

    return au_map


def evaluate_per_au(dataset, stats="all"):
    results = {}

    y = dataset["label"]
    au_map = get_au_columns(dataset)

    for au, cols in au_map.items():

        # --- select AU features ---
        X = dataset[cols]

        # --- apply stat selection ---
        X = select_feature_stats(X, stats)

        # --- run classifier ---
        metrics = run_classifier(X, y)

        results[au] = metrics

    return results

