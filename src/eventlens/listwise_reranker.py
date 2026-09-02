from __future__ import annotations

from collections import Counter

import numpy as np

from eventlens.cross_encoder_event import merge_candidate_names


def build_label_to_article_pairs(
    article_text: str,
    names: list[str],
    descriptions: dict[str, str],
) -> tuple[list[str], list[str]]:
    """Build reranker pairs with the short event intent as query."""

    left = [
        f"事件类型：{name}\n事件定义：{descriptions.get(name, '')}"
        for name in names
    ]
    right = [article_text] * len(names)
    return left, right


def build_candidate_groups(
    svc_scores: np.ndarray,
    classes: list[str],
    recalls,
    *,
    svc_k: int = 5,
    bge_k: int = 5,
    limit: int = 10,
) -> list[list[str]]:
    scores = np.asarray(svc_scores)
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ValueError("svc score shape mismatch")
    order = np.argsort(-scores, axis=1)
    output: list[list[str]] = []
    for index, recall in enumerate(recalls):
        svc_names = [classes[int(j)] for j in order[index, :svc_k]]
        bge_names = [row.event_name for row in recall.candidates[:bge_k]]
        output.append(merge_candidate_names(svc_names, bge_names, limit=limit))
    return output


def ensure_training_positive(
    candidates: list[str],
    truth: str,
    *,
    limit: int = 10,
) -> list[str]:
    values = [str(value) for value in candidates if str(value).strip()]
    if truth in values:
        return values[:limit]
    if limit <= 1:
        return [truth]
    return [truth] + values[: limit - 1]


def inverse_frequency_weights(
    labels: list[str],
    *,
    power: float = 0.5,
) -> dict[str, float]:
    if power < 0:
        raise ValueError("power must be >= 0")
    counts = Counter(labels)
    raw = {label: count ** (-power) for label, count in counts.items()}
    mean = sum(raw.values()) / max(1, len(raw))
    return {label: value / max(mean, 1e-12) for label, value in raw.items()}


def freeze_reranker_last_layers(model, *, last_n: int) -> tuple[int, int]:
    if last_n <= 0:
        raise ValueError("last_n must be positive")
    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone = getattr(model, "roberta", None) or getattr(model, "xlm_roberta", None)
    layers = getattr(getattr(backbone, "encoder", None), "layer", None)
    if layers is None:
        raise ValueError("unsupported reranker backbone")
    for layer in list(layers)[-min(last_n, len(layers)) :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    classifier = getattr(model, "classifier", None) or getattr(model, "score", None)
    if classifier is not None:
        for parameter in classifier.parameters():
            parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total
