import numpy as np


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def safe_divide(a, b):
    if b == 0:
        return None
    return a / b


def normalize_distance(distance, scale):
    return safe_divide(distance, scale)


def baseline_normalize(values, baseline_indices):
    values = np.array(values)

    if baseline_indices is None or len(values[baseline_indices]) == 0:
        return values

    baseline = np.nanmean(values[baseline_indices])
    return values - baseline


def get_point(landmarks, name):
    return landmarks.get(name, None)


def compute_distance_feature(landmarks, pair, scale=None):
    p1_name, p2_name = pair

    p1 = get_point(landmarks, p1_name)
    p2 = get_point(landmarks, p2_name)

    if p1 is None or p2 is None:
        return None

    d = euclidean_distance(p1, p2)

    if scale is not None and scale > 0:
        d = d / scale

    return d


def compute_ratio(val1, val2):
    if val1 is None or val2 is None:
        return None
    if val2 == 0:
        return None
    return val1 / val2


def compute_scale(landmarks):
    p1 = get_point(landmarks, "Eyes_MidPoint")
    p2 = get_point(landmarks, "NostrilsTop_Centre")

    if p1 is None or p2 is None:
        return None

    return euclidean_distance(p1, p2)