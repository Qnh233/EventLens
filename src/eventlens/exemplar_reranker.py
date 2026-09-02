from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def topk_exemplar_scores(
    train_vectors: np.ndarray,
    train_labels: list[str],
    query_vectors: np.ndarray,
    classes: list[str],
    *,
    top_k: int,
) -> np.ndarray:
    """每个类别保留多个真实 Gold exemplar，用 top-k 相似度均值表示多模态类别。"""

    if len(train_vectors) != len(train_labels):
        raise ValueError("train vector/label count mismatch")
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")
    train = normalize_vectors(train_vectors)
    query = normalize_vectors(query_vectors)
    scores = np.full((len(query), len(classes)), -1.0, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=object)
    for class_index, label in enumerate(classes):
        class_vectors = train[labels == label]
        if len(class_vectors) == 0:
            raise ValueError(f"class has no exemplars: {label}")
        similarities = query @ class_vectors.T
        k = min(top_k, similarities.shape[1])
        if k == similarities.shape[1]:
            selected = similarities
        else:
            selected = np.partition(similarities, -k, axis=1)[:, -k:]
        scores[:, class_index] = selected.mean(axis=1)
    return scores
