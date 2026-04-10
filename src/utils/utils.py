import numpy as np


def euclidean_distance(p1, p2):
    """
    Compute Euclidean distance between two (x, y) points.
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def safe_divide(a, b):
    """
    Safe division to avoid division by zero.
    """
    return a / b if b != 0 else 0.0


def compute_scale(landmarks):
    """
    Compute normalization scale using reference points.
    You can change these points later if needed.
    """
    return euclidean_distance(
        landmarks["eye_center"],
        landmarks["nose"]
    )


def normalize_distance(distance, scale):
    """
    Normalize a distance by facial scale.
    """
    return safe_divide(distance, scale)


def baseline_normalize(values, baseline_indices):
    """
    Subtract baseline mean from a time series.

    values: list or np.array
    baseline_indices: list or slice of indices (e.g. pre-event frames)
    """
    values = np.array(values)

    if len(baseline_indices) == 0:
        return values  # no normalization

    baseline = np.mean(values[baseline_indices])
    return values - baseline