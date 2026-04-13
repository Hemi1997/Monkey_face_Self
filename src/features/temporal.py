import numpy as np
import pandas as pd
from src.features.geometry import row_to_landmarks, compute_all_features



def extract_trial_signals(
    trial_row,
    csv_pd,
    au_config,
    fps,
    baseline_window,
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

    for frame in frames:

        # --- safe frame access ---
        if frame not in csv_pd.index:
            continue

        row = csv_pd.loc[frame]

        # --- landmarks (NO filtering here) ---
        landmarks = row_to_landmarks(row)

        # --- geometry ---
        features, meta = compute_all_features(
            landmarks,
            au_config,
            likelihood_threshold
        )

        # --- initialize once ---
        if not trial_features:
            for k in features.keys():
                trial_features[k] = []

        # --- append safely ---
        for k in trial_features.keys():
            trial_features[k].append(features.get(k, np.nan))

        # --- store meta once ---
        if not feature_meta:
            feature_meta = meta

    # --- baseline indices ---
    baseline_indices = [
        i for i, f in enumerate(frames)
        if baseline_start <= f < baseline_end
    ]

    return trial_features, feature_meta, baseline_indices, frames



def process_trial_signals(
    trial_features,
    feature_meta,
    baseline_indices,
    use_baseline=True
):
    """
    Process raw signals:

    WITH baseline:
        → baseline normalize
        → direction alignment
        → activation (>= 0)

    WITHOUT baseline:
        → use absolute signal (no direction, no clipping)

    Returns:
    - processed signals (dict)
    """

    processed = {}

    for name, values in trial_features.items():

        signal = np.array(values, dtype=float)

        if use_baseline and len(baseline_indices) > 0:

            baseline = np.nanmean(signal[baseline_indices])

            if not np.isnan(baseline):
                signal = signal - baseline

            # --- direction alignment ---
            direction = feature_meta.get(name, "neutral")

            if direction == "increase":
                aligned = signal
            elif direction == "decrease":
                aligned = -signal
            else:
                aligned = signal

            # --- activation (keep only expression strength) ---
            aligned = np.maximum(aligned, 0)


        else:
            aligned = signal

        processed[name] = aligned

    return processed



def compute_temporal_features(processed_signals, fps):
    """
    Extract statistical + temporal features from signals.
    Fully NaN-safe (no interpolation).
    """

    features = {}

    for name, signal in processed_signals.items():

        signal = np.array(signal)

        # skip completely invalid signals
        if len(signal) == 0 or np.all(np.isnan(signal)):
            continue

        # -------------------------
        # Core statistics
        # -------------------------
        features[f"{name}__mean"] = np.nanmean(signal)
        features[f"{name}__std"] = np.nanstd(signal)
        features[f"{name}__max"] = np.nanmax(signal)
        features[f"{name}__min"] = np.nanmin(signal)
        features[f"{name}__range"] = np.nanmax(signal) - np.nanmin(signal)

        # -------------------------
        # Dynamics (velocity)
        # -------------------------
        velocity = np.diff(signal)
        velocity = velocity[~np.isnan(velocity)]

        if len(velocity) > 0:
            features[f"{name}__max_vel"] = np.max(velocity)
            features[f"{name}__min_vel"] = np.min(velocity)

        # -------------------------
        # Timing
        # -------------------------
        if not np.all(np.isnan(signal)):
            peak_idx = np.nanargmax(signal)
            features[f"{name}__time_to_peak"] = peak_idx / fps

        # -------------------------
        # Duration above threshold
        # -------------------------
        peak = np.nanmax(signal)

        if peak > 0:
            threshold = 0.5 * peak
            valid = signal[~np.isnan(signal)]

            features[f"{name}__duration"] = np.sum(valid > threshold) / fps
        else:
            features[f"{name}__duration"] = 0.0

        # -------------------------
        # Area Under Curve (AUC)
        # -------------------------
        clean_signal = np.nan_to_num(signal)
        features[f"{name}__auc"] = np.trapz(clean_signal)

    return features


def process_trial(
    trial_row,
    csv_pd,
    au_config,
    fps,
    baseline_window,
    likelihood_threshold
):
    # --- extract raw signals ---
    trial_features, feature_meta, baseline_indices, frames = extract_trial_signals(
        trial_row,
        csv_pd,
        au_config,
        fps,
        likelihood_threshold,
        baseline_window=0
    )

    # --- decide baseline usage ---
    use_baseline = baseline_window > 0

    # --- process signals ---
    processed = process_trial_signals(
        trial_features,
        feature_meta,
        baseline_indices,
        use_baseline=use_baseline
    )

    # --- extract features ---
    features = compute_temporal_features(processed, fps)

    return features


def build_datase_flatten(
    trial_df,
    csv_pd,
    au_config,
    fps,
    likelihood_threshold,
    baseline_window=0
):
    all_rows = []

    for _, trial_row in trial_df.iterrows():

        # --- extract features ---
        feats = process_trial(
            trial_row,
            csv_pd,
            au_config,
            fps,
            baseline_window,
            likelihood_threshold
        )

        # --- build full row ---
        row = {}

        # =========================
        # METADATA
        # =========================
        row["trial_number"] = trial_row.get("trial_number", None)
        row["event_frame"] = trial_row.get("event_frame", None)

        # =========================
        # LABEL
        # =========================
        row["label"] = trial_row.get("trial_info", None)

        # =========================
        # FEATURES
        # =========================
        row.update(feats)

        # =========================
        # OPTIONAL: feature quality
        # =========================
        row["n_features"] = len(feats)

        all_rows.append(row)

    # --- convert to DataFrame ---
    dataset = pd.DataFrame(all_rows)

    return dataset



def dataset_to_long_format(dataset):
    """
    Convert flat dataset into structured long format with:
    - AU
    - type (feature / ratio)
    - feature name
    - stat
    """

    df = dataset.copy()

    # -------------------------
    # Metadata columns
    # -------------------------
    meta_cols = [
        c for c in df.columns
        if c in ["trial_number", "event_frame","label"]
    ]

    feature_cols = [c for c in df.columns if c not in meta_cols]

    # -------------------------
    # Melt
    # -------------------------
    long_df = df.melt(
        id_vars=meta_cols,
        value_vars=feature_cols,
        var_name="feature_full",
        value_name="value"
    )

    # -------------------------
    # Split parts
    # -------------------------
    parts = long_df["feature_full"].str.split("__", expand=True)

    # -------------------------
    # AU
    # -------------------------
    long_df["AU"] = parts[0]

    # -------------------------
    # TYPE (feature vs ratio)
    # -------------------------
    long_df["type"] = parts[1].apply(
        lambda x: "ratio" if x == "ratio" else "feature"
    )

    # -------------------------
    # FEATURE NAME
    # -------------------------
    def extract_feature(row_parts):
        if row_parts[1] == "ratio":
            # skip "ratio" keyword
            return "__".join(row_parts[2:-1])
        else:
            return "__".join(row_parts[1:-1])

    long_df["feature"] = parts.apply(extract_feature, axis=1)

    # -------------------------
    # STAT
    # -------------------------
    long_df["stat"] = parts.iloc[:, -1]

    # -------------------------
    # Cleanup
    # -------------------------
    long_df = long_df.drop(columns=["feature_full"])

    return long_df