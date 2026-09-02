from __future__ import annotations

import numpy as np


def build_rocchio_prototypes(
    features,
    labels: list[str],
    classes: list[str],
    *,
    negative_weight: float,
) -> np.ndarray:
    """在 TF-IDF 空间构造 class centroid - beta * non-class centroid。"""

    if features.shape[0] != len(labels):
        raise ValueError("feature/label count mismatch")
    if negative_weight < 0.0:
        raise ValueError("negative_weight must be non-negative")
    label_array = np.asarray(labels, dtype=object)
    prototypes: list[np.ndarray] = []
    for label in classes:
        positive_mask = label_array == label
        if not np.any(positive_mask):
            raise ValueError(f"missing class in prototype fit: {label}")
        positive = np.asarray(features[positive_mask].mean(axis=0)).ravel()
        if negative_weight > 0.0 and np.any(~positive_mask):
            negative = np.asarray(features[~positive_mask].mean(axis=0)).ravel()
            positive = positive - negative_weight * negative
        norm = float(np.linalg.norm(positive))
        prototypes.append(positive / max(norm, 1e-12))
    return np.asarray(prototypes, dtype=np.float64)


def rocchio_scores(features, prototypes: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features @ prototypes.T, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("rocchio scores must be a matrix")
    return matrix
