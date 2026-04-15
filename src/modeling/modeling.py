import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def get_feature_columns(dataset):
    """
    Return only the actual feature columns.
    Keeps metadata like trial_number, event_frame, n_features out of X.
    """
    return [c for c in dataset.columns if isinstance(c, str) and "__" in c]


def get_feature_matrix(dataset):
    """
    Split dataset into:
    - X: feature-only matrix
    - y: label vector
    """
    feature_cols = get_feature_columns(dataset)

    if "label" not in dataset.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    X = dataset[feature_cols].copy()
    y = dataset["label"].copy()

    return X, y


def run_classifier(
    X,
    y,
    *,
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

    if model_type == "svm":
        model = SVC(kernel="linear", probability=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    acc_scores = []
    f1_scores = []
    auc_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

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
    - string like "mean"
    - list like ["mean", "max"]
    """

    if stats == "all":
        return X.copy()

    if isinstance(stats, str):
        stats = [stats]

    cols = [
        c for c in X.columns
        if any(c.endswith(f"__{s}") for s in stats)
    ]

    return X[cols].copy()


def run_with_feature_selection(
    dataset,
    *,
    stats="all",
    n_splits=5,
    random_state=42,
    model_type="svm"
):
    X, y = get_feature_matrix(dataset)
    X = select_feature_stats(X, stats)

    return run_classifier(
        X,
        y,
        model_type=model_type,
        n_splits=n_splits,
        random_state=random_state
    )


def get_au_columns(dataset):
    """
    Map AU name -> feature columns for that AU.
    Uses only feature columns, not metadata.
    """
    au_map = {}
    feature_cols = get_feature_columns(dataset)

    for col in feature_cols:
        au = col.split("__")[0]
        au_map.setdefault(au, []).append(col)

    return au_map


def evaluate_per_au(
    dataset,
    *,
    stats="all",
    n_splits=5,
    random_state=42,
    model_type="svm"
):
    results = {}

    y = dataset["label"]
    au_map = get_au_columns(dataset)

    for au, cols in au_map.items():
        X = dataset[cols].copy()
        X = select_feature_stats(X, stats)

        if X.shape[1] == 0:
            continue

        metrics = run_classifier(
            X,
            y,
            model_type=model_type,
            n_splits=n_splits,
            random_state=random_state
        )

        results[au] = metrics

    return results


def evaluate_au_ablation(
    dataset,
    *,
    stats="all",
    n_splits=5,
    random_state=42,
    model_type="svm"
):
    """
    Returns absolute metrics for:
    - each AU removed one at a time
    - full model in results["full_model"]

    This is the version that matches your one-table plotting style.
    """
    results = {}

    y = dataset["label"]

    X_full, y = get_feature_matrix(dataset)
    X_full = select_feature_stats(X_full, stats)

    full_metrics = run_classifier(
        X_full,
        y,
        n_splits=n_splits,
        random_state=random_state,
        model_type=model_type
    )

    au_map = get_au_columns(dataset)

    for au, cols in au_map.items():
        cols_filtered = [c for c in cols if c in X_full.columns]

        if len(cols_filtered) == 0:
            continue

        X_drop = X_full.drop(columns=cols_filtered, errors="ignore")

        if X_drop.shape[1] == 0:
            continue

        metrics = run_classifier(
            X_drop,
            y,
            n_splits=n_splits,
            random_state=random_state,
            model_type=model_type
        )

        results[au] = metrics

    results["full_model"] = full_metrics
    return results


def group_rfe(
    dataset,
    *,
    stats="all",
    n_splits=5,
    random_state=42,
    model_type="svm"
):
    y = dataset["label"]

    X, y = get_feature_matrix(dataset)
    X = select_feature_stats(X, stats)

    au_map = get_au_columns(dataset)

    au_map = {
        au: [c for c in cols if c in X.columns]
        for au, cols in au_map.items()
    }
    au_map = {au: cols for au, cols in au_map.items() if cols}

    remaining_aus = list(au_map.keys())
    history = []

    while len(remaining_aus) > 1:
        scores = {}

        for au in remaining_aus:
            cols_to_remove = au_map[au]

            X_temp = X.drop(columns=cols_to_remove, errors="ignore")

            if X_temp.shape[1] == 0:
                continue

            metrics = run_classifier(
                X_temp,
                y,
                n_splits=n_splits,
                random_state=random_state,
                model_type=model_type
            )

            scores[au] = metrics["auc"]

        if not scores:
            break

        worst_au = min(scores, key=scores.get)

        history.append({
            "removed_au": worst_au,
            "score": scores[worst_au]
        })

        X = X.drop(columns=au_map[worst_au], errors="ignore")
        remaining_aus.remove(worst_au)

    return history


def per_au_results_to_df(results_per_au):
    rows = []
    for au, metrics in results_per_au.items():
        row = {"AU": au}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "auc" in df.columns:
        df = df.sort_values("auc", ascending=False)
    return df


def ablation_results_to_df(results_ablate):
    rows = []

    for au, metrics in results_ablate.items():
        au_name = "FULL_MODEL" if au == "full_model" else au

        row = {"AU": au_name}
        row["accuracy"] = metrics["accuracy"]
        row["f1"] = metrics["f1"]
        row["auc"] = metrics["auc"]
        rows.append(row)

    df = pd.DataFrame(rows)
    if "auc" in df.columns:
        df = df.sort_values("auc", ascending=False)

    return df


def rfe_results_to_df(results_rfe_group):
    return pd.DataFrame(results_rfe_group)