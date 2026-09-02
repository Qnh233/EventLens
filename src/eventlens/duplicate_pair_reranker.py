from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

from eventlens.duplicate_pair_evaluation import (
    BinaryMetrics,
    binary_metrics,
    embed_pair_scores,
    select_threshold,
    stratified_pair_split,
)
from eventlens.duplicate_pairs import DuplicatePair
from eventlens.event_retrieval import EmbeddingClient


class PairFeature(BaseModel):
    pair_id: str
    label: int
    title_similarity: float
    bge_cosine: float
    time_closeness: float

    def vector(self) -> list[float]:
        return [self.title_similarity, self.bge_cosine, self.time_closeness]


class RerankerReport(BaseModel):
    train_count: int
    fit_count: int
    validation_count: int
    external_count: int
    threshold: float
    validation: BinaryMetrics
    external: BinaryMetrics
    bge_external: BinaryMetrics
    external_f1_gain: float
    coefficients: dict[str, float]
    intercept: float
    recommended: bool


class RerankerStabilityReport(BaseModel):
    seeds: list[int]
    runs: list[RerankerReport]
    mean_external_f1: float
    min_external_f1: float
    max_external_f1: float
    mean_external_f1_gain: float
    min_external_f1_gain: float
    threshold_min: float
    threshold_max: float
    recommended: bool


def evaluate_lightweight_reranker(
    train_pairs: list[DuplicatePair],
    external_pairs: list[DuplicatePair],
    embedding_client: EmbeddingClient,
    *,
    bge_threshold: float,
    calibration_ratio: float = 0.5,
    seed: int = 42,
    c: float = 1.0,
    max_iter: int = 300,
    min_external_f1_gain: float = 0.01,
) -> RerankerReport:
    """三特征逻辑回归实验；仅在外部集稳定提升时建议采用。"""

    all_pairs = [*train_pairs, *external_pairs]
    feature_map = {
        row.pair_id: row
        for row in build_pair_features(all_pairs, embedding_client)
    }
    return _evaluate_with_feature_map(
        train_pairs,
        external_pairs,
        feature_map,
        bge_threshold=bge_threshold,
        calibration_ratio=calibration_ratio,
        seed=seed,
        c=c,
        max_iter=max_iter,
        min_external_f1_gain=min_external_f1_gain,
    )


def evaluate_reranker_stability(
    train_pairs: list[DuplicatePair],
    external_pairs: list[DuplicatePair],
    embedding_client: EmbeddingClient,
    *,
    bge_threshold: float,
    seeds: list[int],
    calibration_ratio: float = 0.5,
    c: float = 1.0,
    max_iter: int = 300,
    min_mean_external_f1_gain: float = 0.01,
) -> RerankerStabilityReport:
    if not seeds:
        raise ValueError("seeds 不能为空")
    all_pairs = [*train_pairs, *external_pairs]
    feature_map = {
        row.pair_id: row
        for row in build_pair_features(all_pairs, embedding_client)
    }
    runs = [
        _evaluate_with_feature_map(
            train_pairs,
            external_pairs,
            feature_map,
            bge_threshold=bge_threshold,
            calibration_ratio=calibration_ratio,
            seed=seed,
            c=c,
            max_iter=max_iter,
            min_external_f1_gain=0.0,
        )
        for seed in seeds
    ]
    external_f1 = [run.external.f1 for run in runs]
    gains = [run.external_f1_gain for run in runs]
    thresholds = [run.threshold for run in runs]
    bge_precision = runs[0].bge_external.precision
    recommended = (
        sum(gains) / len(gains) >= min_mean_external_f1_gain
        and min(gains) >= 0.0
        and all(run.external.precision >= bge_precision for run in runs)
    )
    return RerankerStabilityReport(
        seeds=seeds,
        runs=runs,
        mean_external_f1=round(sum(external_f1) / len(external_f1), 6),
        min_external_f1=round(min(external_f1), 6),
        max_external_f1=round(max(external_f1), 6),
        mean_external_f1_gain=round(sum(gains) / len(gains), 6),
        min_external_f1_gain=round(min(gains), 6),
        threshold_min=round(min(thresholds), 6),
        threshold_max=round(max(thresholds), 6),
        recommended=recommended,
    )


def _evaluate_with_feature_map(
    train_pairs: list[DuplicatePair],
    external_pairs: list[DuplicatePair],
    feature_map: dict[str, PairFeature],
    *,
    bge_threshold: float,
    calibration_ratio: float,
    seed: int,
    c: float,
    max_iter: int,
    min_external_f1_gain: float,
) -> RerankerReport:
    fit_pairs, validation_pairs = stratified_pair_split(
        train_pairs, calibration_ratio=calibration_ratio, seed=seed
    )

    calibration_model = _fit_model(
        fit_pairs,
        feature_map,
        seed=seed,
        c=c,
        max_iter=max_iter,
    )
    validation_scores = _predict_scores(
        calibration_model, validation_pairs, feature_map
    )
    threshold, validation_metrics = select_threshold(
        validation_pairs, validation_scores
    )

    final_model = _fit_model(
        train_pairs,
        feature_map,
        seed=seed,
        c=c,
        max_iter=max_iter,
    )
    external_scores = _predict_scores(final_model, external_pairs, feature_map)
    external_metrics = binary_metrics(external_pairs, external_scores, threshold)
    bge_scores = {
        pair.pair_id: feature_map[pair.pair_id].bge_cosine
        for pair in external_pairs
    }
    bge_metrics = binary_metrics(external_pairs, bge_scores, bge_threshold)
    gain = round(external_metrics.f1 - bge_metrics.f1, 6)
    coefficients = dict(
        zip(
            ("title_similarity", "bge_cosine", "time_closeness"),
            (round(float(value), 6) for value in final_model.coef_[0]),
        )
    )
    return RerankerReport(
        train_count=len(train_pairs),
        fit_count=len(fit_pairs),
        validation_count=len(validation_pairs),
        external_count=len(external_pairs),
        threshold=round(threshold, 6),
        validation=validation_metrics,
        external=external_metrics,
        bge_external=bge_metrics,
        external_f1_gain=gain,
        coefficients=coefficients,
        intercept=round(float(final_model.intercept_[0]), 6),
        recommended=(
            gain >= min_external_f1_gain
            and external_metrics.precision >= bge_metrics.precision
        ),
    )


def build_pair_features(
    pairs: list[DuplicatePair], embedding_client: EmbeddingClient
) -> list[PairFeature]:
    bge_scores = embed_pair_scores(pairs, embedding_client)
    return [
        PairFeature(
            pair_id=pair.pair_id,
            label=pair.label,
            title_similarity=pair.title_similarity,
            bge_cosine=bge_scores[pair.pair_id],
            time_closeness=_time_closeness(pair.time_gap_days),
        )
        for pair in pairs
    ]


def write_reranker_report(path: str | Path, report: RerankerReport) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _fit_model(
    pairs: list[DuplicatePair],
    feature_map: dict[str, PairFeature],
    *,
    seed: int,
    c: float,
    max_iter: int,
) -> LogisticRegression:
    model = LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(
        [feature_map[pair.pair_id].vector() for pair in pairs],
        [pair.label for pair in pairs],
    )
    return model


def _predict_scores(
    model: LogisticRegression,
    pairs: list[DuplicatePair],
    feature_map: dict[str, PairFeature],
) -> dict[str, float]:
    probabilities = model.predict_proba(
        [feature_map[pair.pair_id].vector() for pair in pairs]
    )[:, 1]
    return {
        pair.pair_id: round(float(score), 6)
        for pair, score in zip(pairs, probabilities)
    }


def _time_closeness(gap_days: float | None) -> float:
    if gap_days is None:
        return 0.0
    return round(1.0 / (1.0 + max(0.0, gap_days)), 6)
