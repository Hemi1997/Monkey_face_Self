import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def run_with_feature_selection(dataset, stats="all", n_splits= 5, random_state = 42, model_type="svm"):
    X = dataset.drop(columns=["label"], errors="ignore")
    y = dataset["label"]

    X = select_feature_stats(X, stats)

    return run_classifier(X, y,  n_splits, random_state, model_type)


def get_au_columns(dataset):
    au_map = {}

    for col in dataset.columns:
        if "__" not in col:
            continue

        au = col.split("__")[0]
        au_map.setdefault(au, []).append(col)

    return au_map


def evaluate_per_au(dataset, stats="all", n_splits= 5, random_state = 42, model_type="svm"):
    results = {}

    y = dataset["label"]
    au_map = get_au_columns(dataset)

    for au, cols in au_map.items():

        # --- select AU features ---
        X = dataset[cols]

        # --- apply stat selection ---
        X = select_feature_stats(X, stats)

        # --- run classifier ---
        metrics = run_classifier(X, y,n_splits, random_state, model_type)

        results[au] = metrics

    return results


def evaluate_au_ablation(
    dataset,
    stats="all",
    n_splits=5,
    random_state=42,
    model_type="svm"
):
    results = {}

    y = dataset["label"]

    # -------------------------
    # FULL MODEL: all AU features
    # -------------------------
    X_full = dataset.drop(columns=["label"], errors="ignore")
    X_full = select_feature_stats(X_full, stats)

    full_metrics = run_classifier(
        X_full,
        y,
        n_splits,
        random_state,
        model_type
    )

    # -------------------------
    # AU MAP
    # -------------------------
    au_map = get_au_columns(dataset)

    for au, cols in au_map.items():

        cols_filtered = [c for c in cols if c in X_full.columns]

        if len(cols_filtered) == 0:
            continue

        # remove this AU
        X_drop = X_full.drop(columns=cols_filtered, errors="ignore")

        if X_drop.shape[1] == 0:
            continue

        metrics = run_classifier(
            X_drop,
            y,
            n_splits,
            random_state,
            model_type
        )

        results[au] = {
            "accuracy_drop": full_metrics["accuracy"] - metrics["accuracy"],
            "f1_drop": full_metrics["f1"] - metrics["f1"],
            "auc_drop": full_metrics["auc"] - metrics["auc"],
        }

    # add the full model at the end
    results["full_model"] = full_metrics

    return results


def group_rfe(dataset, stats="all", n_splits= 5, random_state = 42, model_type="svm"):
    y = dataset["label"]

    X = dataset.drop(columns=["label"], errors="ignore")
    X = select_feature_stats(X, stats)

    au_map = get_au_columns(dataset)

    # keep only columns that survived stat selection
    au_map = {
        au: [c for c in cols if c in X.columns]
        for au, cols in au_map.items()
    }

    # remove empty groups
    au_map = {au: cols for au, cols in au_map.items() if cols}

    remaining_aus = list(au_map.keys())
    history = []

    while len(remaining_aus) > 1:

        scores = {}

        # test removing each AU
        for au in remaining_aus:

            cols_to_remove = au_map[au]

            X_temp = X.drop(
                columns=[c for c in cols_to_remove if c in X.columns]
            )

            if X_temp.shape[1] == 0:
                continue

            metrics = run_classifier(X_temp, y, n_splits, random_state, model_type)

            scores[au] = metrics["auc"]

        # find worst AU (removal improves or least harms performance)
        worst_au = min(scores, key=scores.get)

        history.append({
            "removed_au": worst_au,
            "score": scores[worst_au]
        })

        # remove it permanently
        X = X.drop(columns=au_map[worst_au])
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

        # rename full_model to FULL_MODEL
        if au == "full_model":
            au_name = "FULL_MODEL"
        else:
            au_name = au

        row = {"AU": au_name}

        # just store metrics directly (no drops)
        row["accuracy"] = metrics["accuracy"]
        row["f1"] = metrics["f1"]
        row["auc"] = metrics["auc"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # optional: sort by AUC
    df = df.sort_values("auc", ascending=False)

    return df


def rfe_results_to_df(results_rfe_group):
    return pd.DataFrame(results_rfe_group)