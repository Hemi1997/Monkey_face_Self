import numpy as np 
import pandas as pd
from src.utils import compute_distance_feature, compute_ratio, compute_scale

def row_to_landmarks(row):
    landmarks = {}

    for bodypart in row.index.get_level_values(0).unique():
        try:
            x = row[(bodypart, "x")]
            y = row[(bodypart, "y")]
            likelihood = row[(bodypart, "likelihood")]
        except KeyError:
            continue

        landmarks[bodypart] = {
            "x": x,
            "y": y,
            "likelihood": likelihood
        }

    return landmarks


def make_feature_name(au_name, pair):
    p1, p2 = pair
    return f"{au_name}__{p1}__{p2}"

def make_ratio_name(au_name, pair1, pair2):
    p1a, p1b = pair1
    p2a, p2b = pair2
    return f"{au_name}__ratio__{p1a}__{p1b}__{p2a}__{p2b}"



def compute_au_features(landmarks, au_name, au_config, likelihood_threshold):
    features = {}
    feature_meta = {}

    # --- compute scale once per frame ---
    # scale = compute_scale(landmarks, likelihood_threshold)

    # --- features ---
    for feature in au_config.get("features", []):
        pair = feature["pair"]
        direction = feature.get("direction", None)

        name = make_feature_name(au_name, pair)

        value = compute_distance_feature(
            landmarks,
            pair,
            likelihood_threshold=likelihood_threshold
        )

        features[name] = value
        feature_meta[name] = direction

    # --- ratios ---
    for ratio in au_config.get("ratios", []):
        pair1, pair2 = ratio["pairs"]
        direction = ratio.get("direction", None)

        val1 = compute_distance_feature(
            landmarks,
            pair1,
            likelihood_threshold=likelihood_threshold
        )

        val2 = compute_distance_feature(
            landmarks,
            pair2,
            likelihood_threshold=likelihood_threshold
        )

        name = make_ratio_name(au_name, pair1, pair2)

        features[name] = compute_ratio(val1, val2)
        feature_meta[name] = direction

    return features, feature_meta


def compute_all_features(landmarks, au_config, likelihood_threshold):
    all_features = {}
    all_meta = {}

    for au_name, config in au_config.items():
        feats, meta = compute_au_features(
            landmarks,
            au_name,
            config,
            likelihood_threshold
        )

        all_features.update(feats)
        all_meta.update(meta)

    return all_features, all_meta