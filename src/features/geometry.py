import numpy as np 
import pandas as pd
from src.utils import compute_distance_feature, compute_ratio, compute_scale

def row_to_landmarks(row, likelihood_threshold=0.0):
    """
    Convert a DLC dataframe row into a landmarks dictionary.

    Parameters:
    - row: pandas Series with MultiIndex columns (bodypart, coord)
    - likelihood_threshold: float (optional filtering)

    Returns:
    - dict: {bodypart: (x, y)}
    """

    landmarks = {}

    for bodypart in row.index.get_level_values(0).unique():
        try:
            x = row[(bodypart, "x")]
            y = row[(bodypart, "y")]
            likelihood = row[(bodypart, "likelihood")]
        except KeyError:
            continue  # skip if missing

        # filter low-confidence points
        if likelihood < likelihood_threshold:
            continue

        landmarks[bodypart] = (x, y)

    return landmarks


def make_feature_name(au_name, pair):
    p1, p2 = pair
    return f"{au_name}__{p1}__{p2}"

def make_ratio_name(au_name, pair1, pair2):
    p1a, p1b = pair1
    p2a, p2b = pair2
    return f"{au_name}__ratio__{p1a}__{p1b}__{p2a}__{p2b}"


def compute_au_features(landmarks, au_name, au_config):
    """
    Compute features for a single AU for one frame.
    """

    features = {}

    # optional normalization
    # scale = compute_scale(landmarks)

    # --- distances ---
    for feature in au_config.get("features", []):
        pair = feature["pair"]
        name = make_feature_name(au_name, pair)

        value = compute_distance_feature(
            landmarks,
            pair,
        )

        features[name] = value

    # --- ratios ---
    for ratio in au_config.get("ratios", []):
        pair1, pair2 = ratio["pairs"]
        val1 = compute_distance_feature(landmarks, pair1)
        val2 = compute_distance_feature(landmarks, pair2)

        name = make_ratio_name(au_name, pair1, pair2)

        features[name] = compute_ratio(val1, val2)

    return features


def compute_all_features(landmarks, au_config):
    """
    Compute all AU features for one frame.
    """

    all_features = {}

    for au_name, config in au_config.items():
        au_features = compute_au_features(landmarks, au_name, config)

        all_features.update(au_features)

    return all_features