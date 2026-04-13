import numpy as np
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
    - NO interpolation
    - optional baseline normalization
    - direction alignment
    - activation (>= 0)

    Returns:
    - processed signals (dict)
    """

    processed = {}

    for name, values in trial_features.items():

        signal = np.array(values, dtype=float)

        # --- baseline normalization (NaN-safe) ---
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
        baseline_window,
        likelihood_threshold
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

def build_dataset(
    trial_df,
    csv_pd,
    au_config,
    fps,
    baseline_window,
    likelihood_threshold
):
    all_rows = []

    for _, trial_row in trial_df.iterrows():

        feats = process_trial(
            trial_row,
            csv_pd,
            au_config,
            fps,
            baseline_window,
            likelihood_threshold
        )

        feats["label"] = trial_row["trial_info"]

        all_rows.append(feats)

    return all_rows