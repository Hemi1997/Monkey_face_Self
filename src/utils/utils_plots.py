# utils_plot.py

from __future__ import annotations

from typing import Optional, Union, Sequence

import pandas as pd
import matplotlib.pyplot as plt


def _to_dataframe(obj) -> pd.DataFrame:
    """
    Convert a dict/list/DataFrame into a DataFrame.
    Useful when plot functions accept flexible inputs.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    raise TypeError(f"Unsupported input type: {type(obj)}")


def plot_metric_bars(
    df: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    sort: bool = True,
    ascending: bool = False,
    highlight_label: str = "FULL_MODEL",
    figsize: tuple = (10, 5),
    ax=None,
):
    """
    Generic bar plot for a metric column.

    Expected columns:
    - AU
    - metric (e.g. 'accuracy', 'f1', 'auc')

    If highlight_label exists, it is plotted in a different color.
    """
    df = _to_dataframe(df)

    if "AU" not in df.columns:
        raise ValueError("DataFrame must contain an 'AU' column.")
    if metric not in df.columns:
        raise ValueError(f"DataFrame must contain metric column '{metric}'.")

    plot_df = df[["AU", metric]].dropna()

    if sort:
        plot_df = plot_df.sort_values(metric, ascending=ascending)

    fig, ax = plt.subplots(figsize=figsize) if ax is None else (None, ax)

    labels = plot_df["AU"].astype(str).tolist()
    values = plot_df[metric].tolist()

    colors = ["tab:blue"] * len(labels)
    if highlight_label in labels:
        colors[labels.index(highlight_label)] = "tab:orange"

    ax.bar(labels, values, color=colors)
    ax.set_xlabel("AU")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"{metric.upper()} by AU")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return ax


def plot_per_au_results(
    df: pd.DataFrame,
    metrics: Sequence[str] = ("accuracy", "f1", "auc"),
    sort_by: str = "auc",
    figsize: tuple = (12, 10),
):
    """
    Plot per-AU classification results as one subplot per metric.
    """
    df = _to_dataframe(df)

    required = {"AU", *metrics}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    plot_df = df.copy()
    if sort_by in plot_df.columns:
        plot_df = plot_df.sort_values(sort_by, ascending=False)

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.bar(plot_df["AU"].astype(str), plot_df[metric])
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Per-AU {metric.upper()}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    axes[-1].set_xlabel("AU")
    plt.tight_layout()
    return fig, axes


def plot_ablation_results(
    df: pd.DataFrame,
    metrics: Sequence[str] = ("accuracy", "f1", "auc"),
    sort_by: str = "auc",
    figsize: tuple = (12, 10),
):
    """
    Plot ablation results where each row is an AU and metrics are absolute
    performance values. If a FULL_MODEL row exists, it is shown as a reference line.
    """
    df = _to_dataframe(df)

    required = {"AU", *metrics}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    plot_df = df.copy()

    full_model_row = plot_df[plot_df["AU"] == "FULL_MODEL"]
    plot_df = plot_df[plot_df["AU"] != "FULL_MODEL"]

    if sort_by in plot_df.columns:
        plot_df = plot_df.sort_values(sort_by, ascending=False)

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.bar(plot_df["AU"].astype(str), plot_df[metric], color="tab:blue")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Ablation {metric.upper()}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        if not full_model_row.empty:
            ref = full_model_row.iloc[0][metric]
            ax.axhline(ref, color="tab:orange", linestyle="--", linewidth=2, label="FULL_MODEL")
            ax.legend()

    axes[-1].set_xlabel("AU")
    plt.tight_layout()
    return fig, axes


def plot_group_rfe(
    results: Union[pd.DataFrame, list],
    score_col: str = "score",
    removed_col: str = "removed_au",
    figsize: tuple = (10, 5),
):
    """
    Plot the elimination path for group RFE.

    Accepts:
    - list of dicts
    - DataFrame with columns like ['removed_au', 'score']
    """
    df = _to_dataframe(results)

    if score_col not in df.columns:
        raise ValueError(f"Missing score column '{score_col}'.")
    if removed_col not in df.columns:
        raise ValueError(f"Missing removed AU column '{removed_col}'.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(df) + 1), df[score_col], marker="o")
    ax.set_xlabel("Elimination step")
    ax.set_ylabel(score_col.upper())
    ax.set_title("Group RFE elimination path")
    ax.grid(True, linestyle="--", alpha=0.3)

    # optional labels on points
    for i, au in enumerate(df[removed_col].astype(str), start=1):
        ax.annotate(au, (i, df.iloc[i - 1][score_col]), textcoords="offset points", xytext=(0, 8), ha="center")

    plt.tight_layout()
    return fig, ax


def plot_top_n(
    df: pd.DataFrame,
    metric: str = "auc",
    n: int = 10,
    title: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """
    Plot top-N rows by a chosen metric.
    """
    df = _to_dataframe(df)

    if "AU" not in df.columns:
        raise ValueError("DataFrame must contain an 'AU' column.")
    if metric not in df.columns:
        raise ValueError(f"DataFrame must contain metric column '{metric}'.")

    plot_df = df[["AU", metric]].dropna().sort_values(metric, ascending=False).head(n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(plot_df["AU"].astype(str), plot_df[metric])
    ax.set_xlabel("AU")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Top {n} AUs by {metric.upper()}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, ax