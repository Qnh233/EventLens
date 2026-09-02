from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


def normalized_similarity(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -1.0)
    return similarity


def build_confusion_hard_negative_map(
    texts: list[str],
    labels: list[str],
    classes: list[str],
    anchor_vectors: np.ndarray,
    *,
    model_factory,
    top_k: int = 3,
    semantic_weight: float = 0.35,
    random_state: int = 42,
) -> dict[int, list[int]]:
    """Mine hard negative labels using train-only OOF confusion + label semantics."""

    if len(texts) != len(labels):
        raise ValueError("text/label count mismatch")
    if top_k <= 0 or top_k >= len(classes):
        raise ValueError("top_k must be between 1 and class_count-1")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be within [0, 1]")
    counts = Counter(labels)
    if any(counts[label] < 3 for label in classes):
        raise ValueError("3-fold OOF hard-negative mining requires >=3 samples per class")

    class_to_id = {label: index for index, label in enumerate(classes)}
    prediction = np.empty(len(labels), dtype=object)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    labels_array = np.asarray(labels, dtype=object)
    texts_array = np.asarray(texts, dtype=object)
    for train_idx, val_idx in splitter.split(texts_array, labels_array):
        model = model_factory()
        model.fit(texts_array[train_idx].tolist(), labels_array[train_idx].tolist())
        prediction[val_idx] = model.predict(texts_array[val_idx].tolist())

    matrix = confusion_matrix(labels, prediction.tolist(), labels=classes).astype(np.float64)
    row_totals = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    confusion_rate = matrix / row_totals
    np.fill_diagonal(confusion_rate, 0.0)

    semantic = normalized_similarity(anchor_vectors)
    if semantic.shape != confusion_rate.shape:
        raise ValueError("anchor vector count must match classes")
    semantic = (semantic + 1.0) / 2.0
    np.fill_diagonal(semantic, 0.0)
    combined = (1.0 - semantic_weight) * confusion_rate + semantic_weight * semantic

    output: dict[int, list[int]] = {}
    for label, class_id in class_to_id.items():
        order = np.argsort(-combined[class_id])
        negatives = [int(index) for index in order if int(index) != class_id][:top_k]
        output[class_id] = negatives
    return output
