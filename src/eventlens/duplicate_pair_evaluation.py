from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from pydantic import BaseModel

from eventlens.duplicate_pairs import DuplicatePair
from eventlens.event_retrieval import EmbeddingClient


class BinaryMetrics(BaseModel):
    threshold: float
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


class MethodEvaluation(BaseModel):
    method: str
    calibration: BinaryMetrics
    validation: BinaryMetrics
    positive_score_mean: float
    negative_score_mean: float


class DuplicatePairBenchmark(BaseModel):
    pair_count: int
    calibration_count: int
    validation_count: int
    calibration_ratio: float
    title_similarity: MethodEvaluation
    bge_cosine: MethodEvaluation
    preferred_method: str
    validation_f1_gain: float


class ExternalDuplicatePairEvaluation(BaseModel):
    pair_count: int
    title_similarity: BinaryMetrics
    bge_cosine: BinaryMetrics
    preferred_method: str
    external_f1_gain: float


def load_duplicate_pairs_jsonl(path: str | Path) -> list[DuplicatePair]:
    pairs: list[DuplicatePair] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                pairs.append(DuplicatePair.model_validate_json(line))
    return pairs


def benchmark_duplicate_pairs(
    pairs: list[DuplicatePair],
    embedding_client: EmbeddingClient,
    *,
    calibration_ratio: float = 0.5,
    seed: int = 42,
) -> DuplicatePairBenchmark:
    if len(pairs) < 4:
        raise ValueError("至少需要 4 个文章对进行校准和验证")
    if not 0.1 <= calibration_ratio <= 0.9:
        raise ValueError("calibration_ratio 必须位于 [0.1, 0.9]")
    labels = {pair.label for pair in pairs}
    if labels != {0, 1}:
        raise ValueError("评测数据必须同时包含正样本和负样本")

    calibration, validation = stratified_pair_split(
        pairs, calibration_ratio=calibration_ratio, seed=seed
    )
    title_scores = {pair.pair_id: pair.title_similarity for pair in pairs}
    bge_scores = embed_pair_scores(pairs, embedding_client)

    title_evaluation = _evaluate_method(
        "title_similarity", pairs, calibration, validation, title_scores
    )
    bge_evaluation = _evaluate_method(
        "bge_cosine", pairs, calibration, validation, bge_scores
    )
    preferred = max(
        (title_evaluation, bge_evaluation),
        key=lambda item: (
            item.validation.f1,
            item.validation.precision,
            item.validation.accuracy,
        ),
    ).method
    return DuplicatePairBenchmark(
        pair_count=len(pairs),
        calibration_count=len(calibration),
        validation_count=len(validation),
        calibration_ratio=calibration_ratio,
        title_similarity=title_evaluation,
        bge_cosine=bge_evaluation,
        preferred_method=preferred,
        validation_f1_gain=round(
            bge_evaluation.validation.f1 - title_evaluation.validation.f1, 6
        ),
    )


def evaluate_external_duplicate_pairs(
    pairs: list[DuplicatePair],
    embedding_client: EmbeddingClient,
    *,
    title_threshold: float,
    bge_threshold: float,
) -> ExternalDuplicatePairEvaluation:
    if {pair.label for pair in pairs} != {0, 1}:
        raise ValueError("外部评测数据必须同时包含正样本和负样本")
    title_scores = {pair.pair_id: pair.title_similarity for pair in pairs}
    bge_scores = embed_pair_scores(pairs, embedding_client)
    title_metrics = binary_metrics(pairs, title_scores, title_threshold)
    bge_metrics = binary_metrics(pairs, bge_scores, bge_threshold)
    preferred = max(
        (("title_similarity", title_metrics), ("bge_cosine", bge_metrics)),
        key=lambda item: (item[1].f1, item[1].precision, item[1].accuracy),
    )[0]
    return ExternalDuplicatePairEvaluation(
        pair_count=len(pairs),
        title_similarity=title_metrics,
        bge_cosine=bge_metrics,
        preferred_method=preferred,
        external_f1_gain=round(bge_metrics.f1 - title_metrics.f1, 6),
    )


def _evaluate_method(
    method: str,
    all_pairs: list[DuplicatePair],
    calibration: list[DuplicatePair],
    validation: list[DuplicatePair],
    scores: dict[str, float],
) -> MethodEvaluation:
    threshold, calibration_metrics = select_threshold(calibration, scores)
    validation_metrics = binary_metrics(validation, scores, threshold)
    positive_scores = [scores[pair.pair_id] for pair in all_pairs if pair.label == 1]
    negative_scores = [scores[pair.pair_id] for pair in all_pairs if pair.label == 0]
    return MethodEvaluation(
        method=method,
        calibration=calibration_metrics,
        validation=validation_metrics,
        positive_score_mean=round(mean(positive_scores), 6),
        negative_score_mean=round(mean(negative_scores), 6),
    )


def select_threshold(
    pairs: list[DuplicatePair], scores: dict[str, float]
) -> tuple[float, BinaryMetrics]:
    values = sorted({scores[pair.pair_id] for pair in pairs})
    epsilon = 1e-9
    candidates = [values[0] - epsilon, *values, values[-1] + epsilon]
    evaluations = [binary_metrics(pairs, scores, threshold) for threshold in candidates]
    best = max(
        evaluations,
        key=lambda item: (item.f1, item.precision, item.accuracy, item.threshold),
    )
    return best.threshold, best


def binary_metrics(
    pairs: list[DuplicatePair], scores: dict[str, float], threshold: float
) -> BinaryMetrics:
    tp = fp = tn = fn = 0
    for pair in pairs:
        predicted = int(scores[pair.pair_id] >= threshold)
        if pair.label == 1 and predicted == 1:
            tp += 1
        elif pair.label == 0 and predicted == 1:
            fp += 1
        elif pair.label == 0 and predicted == 0:
            tn += 1
        else:
            fn += 1
    total = len(pairs) or 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        threshold=round(threshold, 6),
        sample_count=len(pairs),
        accuracy=round((tp + tn) / total, 6),
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
    )


def stratified_pair_split(
    pairs: list[DuplicatePair], *, calibration_ratio: float, seed: int
) -> tuple[list[DuplicatePair], list[DuplicatePair]]:
    calibration: list[DuplicatePair] = []
    validation: list[DuplicatePair] = []
    for label in (0, 1):
        rows = [pair for pair in pairs if pair.label == label]
        rows.sort(key=lambda pair: _stable_rank(pair.pair_id, seed))
        cut = max(1, min(len(rows) - 1, round(len(rows) * calibration_ratio)))
        calibration.extend(rows[:cut])
        validation.extend(rows[cut:])
    calibration.sort(key=lambda pair: pair.pair_id)
    validation.sort(key=lambda pair: pair.pair_id)
    return calibration, validation


def _stable_rank(pair_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode("utf-8")).hexdigest()


def embed_pair_scores(
    pairs: list[DuplicatePair], embedding_client: EmbeddingClient
) -> dict[str, float]:
    unique_texts = list(dict.fromkeys(
        text for pair in pairs for text in (pair.left_text, pair.right_text)
    ))
    vectors = embedding_client.embed(unique_texts)
    if len(vectors) != len(unique_texts):
        raise RuntimeError("embedding 返回数量与输入文本数量不一致")
    by_text = dict(zip(unique_texts, vectors))
    return {
        pair.pair_id: round(
            _cosine_similarity(by_text[pair.left_text], by_text[pair.right_text]), 6
        )
        for pair in pairs
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding 维度不一致")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def write_benchmark_report(path: str | Path, benchmark: BaseModel) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(benchmark.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
