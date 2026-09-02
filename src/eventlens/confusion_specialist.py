from __future__ import annotations

from collections import Counter


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError("confusion pair requires two distinct labels")
    return tuple(sorted((str(left), str(right))))


def confusion_pair_counts(
    truth: list[str],
    predictions: list[str],
) -> dict[tuple[str, str], int]:
    if len(truth) != len(predictions):
        raise ValueError("truth/prediction count mismatch")
    counts = Counter(
        canonical_pair(gold, pred)
        for gold, pred in zip(truth, predictions)
        if gold != pred
    )
    return dict(counts)


def select_confusion_pairs(
    counts: dict[tuple[str, str], int],
    *,
    minimum_count: int,
) -> set[tuple[str, str]]:
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    return {
        canonical_pair(left, right)
        for (left, right), count in counts.items()
        if int(count) >= minimum_count
    }
