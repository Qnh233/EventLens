from __future__ import annotations

import numpy as np


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("vectors must be a 2D matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def diverse_balanced_round_robin(
    eligible: np.ndarray,
    group_keys: list[object],
    margins: np.ndarray,
    vectors: np.ndarray,
    count: int,
    *,
    diversity_weight: float = 0.5,
    duplicate_groups: list[str] | None = None,
) -> np.ndarray:
    """类别轮转 + 组内不确定性/表示多样性，且可去除同源重复样本。"""

    if not 0.0 <= diversity_weight <= 1.0:
        raise ValueError("diversity_weight must be within [0, 1]")
    if len(group_keys) != len(margins) or len(group_keys) != len(vectors):
        raise ValueError("acquisition inputs must have equal length")
    if duplicate_groups is not None and len(duplicate_groups) != len(group_keys):
        raise ValueError("duplicate_groups length mismatch")
    if count <= 0:
        return np.empty(0, dtype=np.int64)

    normalized = normalize_vectors(vectors)
    groups: dict[object, list[int]] = {}
    for index in np.asarray(eligible, dtype=np.int64).tolist():
        groups.setdefault(group_keys[index], []).append(index)

    selected: list[int] = []
    selected_by_group: dict[object, list[int]] = {key: [] for key in groups}
    used_duplicate_groups: set[str] = set()
    ordered_groups = sorted(groups, key=lambda key: str(key))

    def next_index(key: object) -> int | None:
        candidates = [
            index
            for index in groups[key]
            if index not in selected_by_group[key]
            and (
                duplicate_groups is None
                or duplicate_groups[index] not in used_duplicate_groups
            )
        ]
        if not candidates:
            return None
        prior = selected_by_group[key]
        if not prior:
            return min(candidates, key=lambda index: (float(margins[index]), index))

        margin_values = np.asarray([margins[index] for index in groups[key]], dtype=np.float64)
        margin_min = float(margin_values.min())
        margin_span = max(float(margin_values.max() - margin_min), 1e-12)
        prior_vectors = normalized[prior]
        best_index = None
        best_score = -np.inf
        for index in candidates:
            max_similarity = float((normalized[index] @ prior_vectors.T).max())
            min_distance = 1.0 - max_similarity
            margin_norm = (float(margins[index]) - margin_min) / margin_span
            score = diversity_weight * min_distance - (1.0 - diversity_weight) * margin_norm
            if score > best_score or (score == best_score and (best_index is None or index < best_index)):
                best_index = index
                best_score = score
        return best_index

    while len(selected) < count:
        added = False
        for key in ordered_groups:
            index = next_index(key)
            if index is None:
                continue
            selected.append(index)
            selected_by_group[key].append(index)
            if duplicate_groups is not None:
                used_duplicate_groups.add(duplicate_groups[index])
            added = True
            if len(selected) >= count:
                break
        if not added:
            break
    return np.asarray(selected, dtype=np.int64)
