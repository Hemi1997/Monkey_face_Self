import numpy as np


# -------------------------
# Core math
# -------------------------

def euclidean_distance(p1, p2):
    return np.sqrt(
        (p1["x"] - p2["x"])**2 +
        (p1["y"] - p2["y"])**2
    )


def safe_divide(a, b):
    if a is None or b is None:
        return np.nan
    if b == 0 or np.isnan(b):
        return np.nan
    return a / b


# -------------------------
# Landmark access
# -------------------------

def get_point(landmarks, name):
    return landmarks.get(name, None)


# -------------------------
# Distance with likelihood filtering
# -------------------------

def compute_distance_feature(landmarks, pair, scale=None, likelihood_threshold=0.8):
    p1_name, p2_name = pair

    p1 = get_point(landmarks, p1_name)
    p2 = get_point(landmarks, p2_name)

    if p1 is None or p2 is None:
        return np.nan

    # --- likelihood filtering HERE ---
    if (
        p1["likelihood"] < likelihood_threshold or
        p2["likelihood"] < likelihood_threshold
    ):
        return np.nan

    d = euclidean_distance(p1, p2)

    # --- scale normalization ---
    if scale is not None:
        if scale == 0 or np.isnan(scale):
            return np.nan
        d = d / scale

    return d


# -------------------------
# Ratio
# -------------------------

def compute_ratio(val1, val2):
    return safe_divide(val1, val2)


# -------------------------
# Scale (robust)
# -------------------------

def compute_scale(landmarks, likelihood_threshold=0.8):
    p1 = get_point(landmarks, "Eyes_MidPoint")
    p2 = get_point(landmarks, "NostrilsTop_Centre")

    if p1 is None or p2 is None:
        return np.nan

    if (
        p1["likelihood"] < likelihood_threshold or
        p2["likelihood"] < likelihood_threshold
    ):
        return np.nan

    return euclidean_distance(p1, p2)