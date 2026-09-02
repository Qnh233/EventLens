from __future__ import annotations

import numpy as np


def row_rank_scores(scores: np.ndarray) -> np.ndarray:
    """Convert arbitrary classifier scales to comparable [0, 1] rank scores."""

    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("rank fusion requires a 2D score matrix with >=2 classes")
    order = np.argsort(-matrix, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(matrix.shape[0])[:, None]
    ranks[rows, order] = np.arange(matrix.shape[1])[None, :]
    return 1.0 - ranks / float(matrix.shape[1] - 1)


def blend_rank_scores(
    primary_scores: np.ndarray,
    semantic_scores: np.ndarray,
    *,
    semantic_weight: float,
) -> np.ndarray:
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be in [0, 1]")
    primary = row_rank_scores(primary_scores)
    semantic = row_rank_scores(semantic_scores)
    if primary.shape != semantic.shape:
        raise ValueError("score matrices must have the same shape")
    return (1.0 - semantic_weight) * primary + semantic_weight * semantic


def top1_margin(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("margin requires a 2D score matrix with >=2 classes")
    partitioned = np.partition(matrix, -2, axis=1)
    return partitioned[:, -1] - partitioned[:, -2]


def gated_predictions(
    primary_predictions: list[str],
    semantic_predictions: list[str],
    primary_margins: np.ndarray,
    *,
    margin_threshold: float,
) -> list[str]:
    if not (
        len(primary_predictions)
        == len(semantic_predictions)
        == len(primary_margins)
    ):
        raise ValueError("gated fusion inputs must have equal length")
    return [
        semantic if margin <= margin_threshold else primary
        for primary, semantic, margin in zip(
            primary_predictions, semantic_predictions, primary_margins
        )
    ]

