# src/features/timecourse.py
import numpy as np
import pandas as pd

from src.features.geometry import row_to_landmarks


def _get_condition_label(trial_row, condition_map=None):
    """
    Convert trial_info into a readable condition label.
    If trial_info is already a string, keep it.
    If condition_map is provided, map numeric labels.
    """
    raw = trial_row.get("trial_info", None)

    if condition_map is not None:
        return condition_map.get(raw, raw)

    return raw


def _compute_distance_from_landmarks(p1, p2, min_likelihood=0.5):
    """
    Compute Euclidean distance if both landmarks pass likelihood threshold.
    Returns np.nan otherwise.
    """
    if p1 is None or p2 is None:
        return np.nan

    if p1["likelihood"] < min_likelihood or p2["likelihood"] < min_likelihood:
        return np.nan

    return np.sqrt(
        (p1["x"] - p2["x"]) ** 2 +
        (p1["y"] - p2["y"]) ** 2
    )


def _build_relative_frame_window(event_frame, fps, window_ms=(-500, 1000)):
    """
    Build a frame-based window around the event.

    Returns:
        rel_frames: array of relative frame offsets
        rel_times_ms: array of relative times in ms
    """
    start_ms, end_ms = window_ms

    start_rel_frame = int(np.floor(start_ms / 1000.0 * fps))
    end_rel_frame = int(np.ceil(end_ms / 1000.0 * fps))

    rel_frames = np.arange(start_rel_frame, end_rel_frame + 1, 1)
    rel_times_ms = (rel_frames / fps) * 1000.0

    return rel_frames, rel_times_ms


def extract_timecourse_for_trial(
    *,
    trial_row,
    csv_pd,
    au_name,
    au_config,
    fps,
    window_ms=(-500, 1000),
    condition_map=None,
    min_likelihood=0.5
):
    """
    Extract time-course rows for one trial and one AU.

    Output is one row per time point per pair, with:
    - both landmark coordinates
    - pair distance
    - relative time from event
    """
    event_frame = int(trial_row["event_frame"])
    trial_number = trial_row.get("trial_number", None)
    trial_index = trial_row.get("trial_index", None)
    condition = _get_condition_label(trial_row, condition_map=condition_map)

    if au_name not in au_config:
        return []

    au_cfg = au_config[au_name]
    pairs = []

    for feature in au_cfg.get("features", []):
        pair = feature["pair"]
        pairs.append({
            "pair": pair,
            "pair_type": "feature",
            "direction": feature.get("direction", None)
        })

    for ratio in au_cfg.get("ratios", []):
        # for now, keep ratios out of the timecourse plot path unless you want them later
        # you can extend this if needed
        pass

    rel_frames, rel_times_ms = _build_relative_frame_window(
        event_frame=event_frame,
        fps=fps,
        window_ms=window_ms
    )

    rows = []

    for pair_info in pairs:
        p1_name, p2_name = pair_info["pair"]

        for rel_f, rel_t_ms in zip(rel_frames, rel_times_ms):
            frame = event_frame + int(rel_f)

            if frame not in csv_pd.index:
                continue

            row = csv_pd.loc[frame]
            landmarks = row_to_landmarks(row)

            p1 = landmarks.get(p1_name, None)
            p2 = landmarks.get(p2_name, None)

            distance = _compute_distance_from_landmarks(
                p1,
                p2,
                min_likelihood=min_likelihood
            )

            rows.append({
                "trial_index": trial_index,
                "trial_number": trial_number,
                "label": trial_row.get("trial_info", None),
                "condition": condition,
                "event_frame": event_frame,
                "frame": frame,
                "rel_frame": int(rel_f),
                "rel_time_ms": float(rel_t_ms),
                "AU": au_name,
                "pair": f"{p1_name}__{p2_name}",
                "pair_type": pair_info["pair_type"],
                "direction": pair_info["direction"],

                "landmark1_name": p1_name,
                "landmark1_x": np.nan if p1 is None else p1["x"],
                "landmark1_y": np.nan if p1 is None else p1["y"],
                "landmark1_likelihood": np.nan if p1 is None else p1["likelihood"],

                "landmark2_name": p2_name,
                "landmark2_x": np.nan if p2 is None else p2["x"],
                "landmark2_y": np.nan if p2 is None else p2["y"],
                "landmark2_likelihood": np.nan if p2 is None else p2["likelihood"],

                "distance": distance,
            })

    return rows


def build_timecourse_df(
    *,
    trial_df,
    csv_pd,
    au_config,
    fps,
    window_ms=(-500, 1000),
    condition_map=None,
    au_names=None,
    min_likelihood=0.5
):
    """
    Build one long DataFrame for time-course plotting.

    Columns include trial metadata, AU, pair, relative time, both landmark
    coordinates, and Euclidean distance.
    """
    all_rows = []

    if au_names is None:
        au_names = list(au_config.keys())

    for _, trial_row in trial_df.iterrows():
        for au_name in au_names:
            rows = extract_timecourse_for_trial(
                trial_row=trial_row,
                csv_pd=csv_pd,
                au_name=au_name,
                au_config=au_config,
                fps=fps,
                window_ms=window_ms,
                condition_map=condition_map,
                min_likelihood=min_likelihood
            )
            all_rows.extend(rows)

    if len(all_rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values(
        ["trial_number", "AU", "pair", "rel_frame"],
        ascending=True
    ).reset_index(drop=True)

    return df


def summarize_timecourse_df(timecourse_df):
    """
    Convenience helper: average across trials for plotting.
    """
    if timecourse_df.empty:
        return timecourse_df

    summary = (
        timecourse_df
        .groupby(["condition", "AU", "pair", "rel_frame", "rel_time_ms"], as_index=False)
        .agg(
            landmark1_x_mean=("landmark1_x", "mean"),
            landmark1_y_mean=("landmark1_y", "mean"),
            landmark2_x_mean=("landmark2_x", "mean"),
            landmark2_y_mean=("landmark2_y", "mean"),
            distance_mean=("distance", "mean"),
            n_trials=("trial_number", "nunique")
        )
    )

    return summary