from __future__ import annotations

from collections import Counter

import numpy as np


def merge_event_candidates(*groups: list[str], limit: int | None = None) -> list[str]:
    """Deduplicate ranked candidate groups while preserving source priority."""

    output: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            label = str(value).strip()
            if not label or label in seen:
                continue
            output.append(label)
            seen.add(label)
            if limit is not None and len(output) >= limit:
                return output
    return output


def build_training_group(
    truth: str,
    *,
    svc_candidates: list[str],
    bge_candidates: list[str],
    hard_candidates: list[str],
    max_negatives: int,
) -> list[str]:
    """Build one group with Gold positive + automatically mined hard negatives."""

    if max_negatives <= 0:
        raise ValueError("max_negatives must be positive")
    negatives = [
        label
        for label in merge_event_candidates(
            svc_candidates,
            bge_candidates,
            hard_candidates,
        )
        if label != truth
    ][:max_negatives]
    if not negatives:
        raise ValueError("training group requires at least one negative")
    return [truth, *negatives]


def class_sample_weights(
    labels: list[str],
    *,
    power: float,
) -> np.ndarray:
    if power < 0:
        raise ValueError("power must be >= 0")
    if not labels:
        raise ValueError("labels must not be empty")
    counts = Counter(labels)
    raw = np.asarray(
        [(len(labels) / (len(counts) * counts[label])) ** power for label in labels],
        dtype=np.float32,
    )
    return raw / max(float(raw.mean()), 1e-12)


def freeze_reranker_except_last_layers(model, *, last_n: int) -> tuple[int, int]:
    """Keep the scoring head and only the final encoder blocks trainable."""

    if last_n <= 0:
        raise ValueError("last_n must be >= 1")
    for parameter in model.parameters():
        parameter.requires_grad = False

    encoder = getattr(model, "roberta", None) or getattr(model, "bert", None) or getattr(model, "xlm_roberta", None)
    layers = getattr(getattr(encoder, "encoder", None), "layer", None)
    if layers is None:
        raise ValueError("unsupported reranker architecture: encoder.layer not found")
    layers = list(layers)
    for layer in layers[-min(last_n, len(layers)) :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    # Sequence-classification heads vary across model families; train any head
    # parameter that lives outside the base encoder.
    encoder_prefixes = ("roberta.", "bert.", "xlm_roberta.")
    for name, parameter in model.named_parameters():
        if not name.startswith(encoder_prefixes):
            parameter.requires_grad = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return int(trainable), int(total)


def choose_group_predictions(
    candidate_groups: list[list[str]],
    score_groups: list[list[float]],
) -> list[str]:
    if len(candidate_groups) != len(score_groups):
        raise ValueError("candidate/score group count mismatch")
    output: list[str] = []
    for candidates, scores in zip(candidate_groups, score_groups):
        if not candidates or len(candidates) != len(scores):
            raise ValueError("every group needs equally sized candidates and scores")
        output.append(candidates[int(np.argmax(np.asarray(scores, dtype=np.float64)))])
    return output
