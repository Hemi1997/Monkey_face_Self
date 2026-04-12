import numpy as np
from src.features.geometry import row_to_landmarks, compute_all_features


def extract_trial_signals(
    trial_row,
    csv_pd,
    au_config,
    fps,
    baseline_window,
    likelihood_threshold=0.0
):
    """
    Extract time-series feature signals for a single trial.

    Parameters:
    - trial_row: row from trial dataframe
    - csv_pd: DLC dataframe (frame-level)
    - au_config: AU definitions
    - fps: frames per second
    - baseline_window: seconds before event
    - likelihood_threshold: filter weak landmarks

    Returns:
    - trial_features: dict {feature_name: [values]}
    - feature_meta: dict {feature_name: direction}
    - baseline_indices: indices of baseline frames (relative)
    - frames: list of frame numbers used
    """

    # --- core frame definitions ---
    event_frame = int(trial_row["event_frame"])
    end_frame = int(trial_row["end_frame"])

    # --- baseline window ---
    baseline_frames = int(baseline_window * fps)

    baseline_start = max(0, event_frame - baseline_frames)
    baseline_end = event_frame

    # --- full window (baseline + response) ---
    frames = list(range(baseline_start, end_frame))

    trial_features = {}
    feature_meta = {}

    # --- extract features frame-by-frame ---
    for frame in frames:

        row = csv_pd.loc[frame]

        # landmarks
        landmarks = row_to_landmarks(row, likelihood_threshold)

        # geometry features
        features, meta = compute_all_features(landmarks, au_config)

        # store time series
        for k, v in features.items():
            trial_features.setdefault(k, []).append(v)

        # store meta once
        if not feature_meta:
            feature_meta = meta

    # --- baseline indices (relative to time series) ---
    baseline_indices = [
        i for i, f in enumerate(frames)
        if baseline_start <= f < baseline_end
    ]

    return trial_features, feature_meta, baseline_indices, frames