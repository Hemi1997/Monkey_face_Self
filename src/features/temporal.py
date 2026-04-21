import numpy as np
import pandas as pd
from src.features.geometry import row_to_landmarks, compute_all_features


def extract_trial_signals(
    *,
    trial_row,
    csv_pd,
    au_config,
    fps=30,
    baseline_window=0,
    likelihood_threshold=0.8
):
    event_frame = int(trial_row["event_frame"])
    end_frame = int(trial_row["end_frame"])

    baseline_frames = int(baseline_window * fps)

    baseline_start = max(0, event_frame - baseline_frames)
    baseline_end = event_frame

    frames = list(range(baseline_start, end_frame))

    trial_features = {}
    feature_meta = {}
    kept_frames = []

    for frame in frames:
        if frame not in csv_pd.index:
            continue

        row = csv_pd.loc[frame]
        landmarks = row_to_landmarks(row)

        features, meta = compute_all_features(
            landmarks=landmarks,
            au_config=au_config,
            likelihood_threshold=likelihood_threshold
        )

        if not trial_features:
            for k in features.keys():
                trial_features[k] = []

        for k in trial_features.keys():
            trial_features[k].append(features.get(k, np.nan))

        kept_frames.append(frame)

        if not feature_meta:
            feature_meta = meta

    baseline_indices = [
        i for i, f in enumerate(kept_frames)
        if baseline_start <= f < baseline_end
    ]

    return trial_features, feature_meta, baseline_indices, kept_frames


def process_trial_signals(
    *,
    trial_features,
    feature_meta,
    baseline_indices,
    use_baseline=True
):
    processed = {}

    for name, values in trial_features.items():
        signal = np.array(values, dtype=float)

        if use_baseline and len(baseline_indices) > 0:
            baseline_vals = signal[baseline_indices]
            baseline_vals = baseline_vals[~np.isnan(baseline_vals)]

            if len(baseline_vals) > 0:
                baseline = np.mean(baseline_vals)
                signal = signal - baseline

            direction = feature_meta.get(name, "neutral")

            if direction == "increase":
                aligned = signal
            elif direction == "decrease":
                aligned = -signal
            else:
                aligned = signal

            aligned = np.maximum(aligned, 0)
        else:
            aligned = signal

        processed[name] = aligned

    return processed


def compute_temporal_features(processed_signals, fps):
    features = {}

    for name, signal in processed_signals.items():
        signal = np.array(signal)

        if len(signal) == 0 or np.all(np.isnan(signal)):
            continue

        features[f"{name}__mean"] = np.nanmean(signal)
        features[f"{name}__std"] = np.nanstd(signal)
        features[f"{name}__max"] = np.nanmax(signal)
        features[f"{name}__min"] = np.nanmin(signal)
        features[f"{name}__range"] = np.nanmax(signal) - np.nanmin(signal)

        velocity = np.diff(signal)
        velocity = velocity[~np.isnan(velocity)]

        if len(velocity) > 0:
            features[f"{name}__max_vel"] = np.max(velocity)
            features[f"{name}__min_vel"] = np.min(velocity)

        if not np.all(np.isnan(signal)):
            peak_idx = np.nanargmax(signal)
            features[f"{name}__time_to_peak"] = peak_idx / fps

        clean_signal = np.nan_to_num(signal)
        features[f"{name}__auc"] = np.trapz(clean_signal)

    return features


def process_trial(
    *,
    trial_row,
    csv_pd,
    au_config,
    fps,
    likelihood_threshold,
    baseline_window=0
):
    trial_features, feature_meta, baseline_indices, frames = extract_trial_signals(
        trial_row=trial_row,
        csv_pd=csv_pd,
        au_config=au_config,
        fps=fps,
        baseline_window=baseline_window,
        likelihood_threshold=likelihood_threshold
    )

    use_baseline = baseline_window > 0

    processed = process_trial_signals(
        trial_features=trial_features,
        feature_meta=feature_meta,
        baseline_indices=baseline_indices,
        use_baseline=use_baseline
    )

    features = compute_temporal_features(processed, fps=fps)

    return features


def build_dataset_flatten(
    *,
    trial_df,
    csv_pd,
    au_config,
    fps,
    likelihood_threshold,
    baseline_window=0
):
    all_rows = []

    for _, trial_row in trial_df.iterrows():
        feats = process_trial(
            trial_row=trial_row,
            csv_pd=csv_pd,
            au_config=au_config,
            fps=fps,
            likelihood_threshold=likelihood_threshold,
            baseline_window=baseline_window
        )

        row = {}
        row["trial_number"] = trial_row.get("trial_number", None)
        row["event_frame"] = trial_row.get("event_frame", None)
        row["label"] = trial_row.get("trial_info", None)
        row.update(feats)
        row["n_features"] = len(feats)

        all_rows.append(row)

    dataset = pd.DataFrame(all_rows)
    return dataset


def dataset_to_long_format(dataset):
    df = dataset.copy()

    meta_cols = [
        c for c in df.columns
        if c in ["trial_index", "trial_number", "event_frame", "end_frame", "label", "n_features"]
    ]

    feature_cols = [
        c for c in df.columns
        if c not in meta_cols and isinstance(c, str) and "__" in c
    ]

    long_df = df.melt(
        id_vars=meta_cols,
        value_vars=feature_cols,
        var_name="feature_full",
        value_name="value"
    )

    parts = long_df["feature_full"].str.split("__", expand=True)
    long_df["AU"] = parts[0]

    def get_type(row_parts):
        if len(row_parts) > 1 and row_parts[1] == "ratio":
            return "ratio"
        return "feature"

    def get_feature(row_parts):
        row_parts = [p for p in row_parts if pd.notna(p)]

        if len(row_parts) < 4:
            return np.nan

        if row_parts[1] == "ratio":
            return "__".join(row_parts[2:-1])
        else:
            return "__".join(row_parts[1:-1])

    def get_stat(row_parts):
        row_parts = [p for p in row_parts if pd.notna(p)]
        if len(row_parts) < 2:
            return np.nan
        return row_parts[-1]

    long_df["type"] = parts.apply(get_type, axis=1)
    long_df["feature"] = parts.apply(get_feature, axis=1)
    long_df["stat"] = parts.apply(get_stat, axis=1)

    long_df = long_df.drop(columns=["feature_full"])
    return long_df