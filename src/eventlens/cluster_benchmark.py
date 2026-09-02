from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from datetime import datetime

from pydantic import BaseModel, Field

from eventlens.cluster import multi_factor_similarity
from eventlens.duplicate_pairs import DuplicateClusterGroup
from eventlens.preprocess import clean_text
from eventlens.schema import ArticleRecord, EventPrediction
from eventlens.semantic_similarity import SemanticPairScorer


class ClusterBenchmarkDataset(BaseModel):
    scope: str
    articles: list[ArticleRecord]
    predictions: list[EventPrediction]
    truth_by_article: dict[str, str]
    article_count: int
    truth_group_count: int
    subject_count: int


class ClusterPairFeature(BaseModel):
    pair_id: str
    left_article_id: str
    right_article_id: str
    rule_score: float
    semantic_score: float | None = None


class ClusterMetrics(BaseModel):
    article_count: int
    predicted_cluster_count: int
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    b_cubed_precision: float
    b_cubed_recall: float
    b_cubed_f1: float
    overmerged_cluster_rate: float
    overmerged_article_rate: float
    split_truth_cluster_rate: float


class ClusterGridResult(BaseModel):
    candidate_threshold: float
    semantic_threshold: float
    top_k: int
    metrics: ClusterMetrics
    semantic_candidate_count: int
    semantic_merged_count: int
    average_semantic_candidates_per_article: float
    pairwise_recall_gain: float
    b_cubed_f1_gain: float
    recommended: bool


class ClusterBenchmarkReport(BaseModel):
    scope: str
    article_count: int
    truth_group_count: int
    subject_count: int
    total_pair_count: int
    retained_rule_pair_count: int
    semantic_scored_pair_count: int
    feature_prepare_seconds: float
    baseline: ClusterMetrics
    grid: list[ClusterGridResult]
    best: ClusterGridResult | None = None
    resource_gate_passed: bool = False


class ClusterWorkloadReport(BaseModel):
    article_count: int
    subject_resolved_count: int
    naive_pair_count: int
    blocked_pair_count: int
    reduction_ratio: float
    candidate_cap: int
    unique_subject_count: int
    missing_time_count: int
    estimated_float32_cache_gb_at_300k: float
    estimated_json_cache_gb_at_300k: float
    requires_candidate_blocking_before_full_scale: bool


class _UnionFind:
    def __init__(self, article_ids: list[str]):
        self.parent = {article_id: article_id for article_id in article_ids}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_cluster_benchmark_dataset(
    groups: list[DuplicateClusterGroup], *, scope: str
) -> ClusterBenchmarkDataset:
    articles: list[ArticleRecord] = []
    predictions: list[EventPrediction] = []
    truth: dict[str, str] = {}
    for group in groups:
        for article in group.articles:
            articles.append(article)
            truth[article.article_id] = group.duplication_id
            predictions.append(
                EventPrediction(
                    article_id=article.article_id,
                    has_event=True,
                    event_type=group.event_label,
                    classifier_confidence=1.0,
                    event_subject=group.subject_name,
                    event_time=article.publish_time,
                    evidence_sentence=(clean_text(article.content) or clean_text(article.title))[:160],
                    extraction_confidence=1.0,
                )
            )
    return ClusterBenchmarkDataset(
        scope=scope,
        articles=articles,
        predictions=predictions,
        truth_by_article=truth,
        article_count=len(articles),
        truth_group_count=len(groups),
        subject_count=len({group.subject_key for group in groups}),
    )


def run_cluster_grid_benchmark(
    dataset: ClusterBenchmarkDataset,
    *,
    cluster_config: dict,
    semantic_scorer: SemanticPairScorer,
    candidate_thresholds: list[float],
    semantic_thresholds: list[float],
    top_ks: list[int],
    minimum_b_cubed_f1_gain: float = 0.02,
    minimum_pairwise_recall_gain: float = 0.05,
) -> ClusterBenchmarkReport:
    started = time.perf_counter()
    features, total_pair_count, semantic_scored = _prepare_features(
        dataset,
        cluster_config=cluster_config,
        semantic_scorer=semantic_scorer,
        minimum_candidate_threshold=min(candidate_thresholds),
        maximum_top_k=max(top_ks),
    )
    prepare_seconds = time.perf_counter() - started
    baseline_mapping = _cluster_mapping(
        dataset,
        features,
        rule_threshold=cluster_config["threshold"],
    )
    baseline = calculate_cluster_metrics(dataset.truth_by_article, baseline_mapping)

    grid: list[ClusterGridResult] = []
    for candidate_threshold, semantic_threshold, top_k in itertools.product(
        candidate_thresholds, semantic_thresholds, top_ks
    ):
        mapping, candidate_count, semantic_merged_count = _semantic_cluster_mapping(
            dataset,
            features,
            rule_threshold=cluster_config["threshold"],
            candidate_threshold=candidate_threshold,
            semantic_threshold=semantic_threshold,
            top_k=top_k,
        )
        metrics = calculate_cluster_metrics(dataset.truth_by_article, mapping)
        recall_gain = metrics.pairwise_recall - baseline.pairwise_recall
        b3_gain = metrics.b_cubed_f1 - baseline.b_cubed_f1
        recommended = (
            metrics.pairwise_precision + 1e-9 >= baseline.pairwise_precision
            and metrics.overmerged_cluster_rate <= baseline.overmerged_cluster_rate + 1e-9
            and (
                b3_gain >= minimum_b_cubed_f1_gain
                or recall_gain >= minimum_pairwise_recall_gain
            )
        )
        grid.append(
            ClusterGridResult(
                candidate_threshold=candidate_threshold,
                semantic_threshold=semantic_threshold,
                top_k=top_k,
                metrics=metrics,
                semantic_candidate_count=candidate_count,
                semantic_merged_count=semantic_merged_count,
                average_semantic_candidates_per_article=round(
                    candidate_count / max(1, dataset.article_count), 6
                ),
                pairwise_recall_gain=round(recall_gain, 6),
                b_cubed_f1_gain=round(b3_gain, 6),
                recommended=recommended,
            )
        )

    eligible = [row for row in grid if row.recommended]
    best = max(
        eligible,
        key=lambda row: (
            row.metrics.b_cubed_f1,
            row.metrics.pairwise_recall,
            -row.average_semantic_candidates_per_article,
        ),
        default=None,
    )
    return ClusterBenchmarkReport(
        scope=dataset.scope,
        article_count=dataset.article_count,
        truth_group_count=dataset.truth_group_count,
        subject_count=dataset.subject_count,
        total_pair_count=total_pair_count,
        retained_rule_pair_count=len(features),
        semantic_scored_pair_count=semantic_scored,
        feature_prepare_seconds=round(prepare_seconds, 4),
        baseline=baseline,
        grid=grid,
        best=best,
        resource_gate_passed=best is not None,
    )


def calculate_cluster_metrics(
    truth_by_article: dict[str, str], predicted_by_article: dict[str, str]
) -> ClusterMetrics:
    intersections: Counter[tuple[str, str]] = Counter()
    truth_sizes: Counter[str] = Counter()
    predicted_sizes: Counter[str] = Counter()
    for article_id, truth_cluster in truth_by_article.items():
        predicted_cluster = predicted_by_article[article_id]
        intersections[(truth_cluster, predicted_cluster)] += 1
        truth_sizes[truth_cluster] += 1
        predicted_sizes[predicted_cluster] += 1

    true_pairs = sum(_choose_two(size) for size in truth_sizes.values())
    predicted_pairs = sum(_choose_two(size) for size in predicted_sizes.values())
    true_positive = sum(_choose_two(size) for size in intersections.values())
    precision = true_positive / predicted_pairs if predicted_pairs else 1.0
    recall = true_positive / true_pairs if true_pairs else 1.0
    pair_f1 = _harmonic(precision, recall)

    b_precision = 0.0
    b_recall = 0.0
    for (truth_cluster, predicted_cluster), size in intersections.items():
        b_precision += size * size / predicted_sizes[predicted_cluster]
        b_recall += size * size / truth_sizes[truth_cluster]
    article_count = len(truth_by_article)
    b_precision /= max(1, article_count)
    b_recall /= max(1, article_count)

    predicted_truth_sets: dict[str, set[str]] = defaultdict(set)
    truth_predicted_sets: dict[str, set[str]] = defaultdict(set)
    for truth_cluster, predicted_cluster in intersections:
        predicted_truth_sets[predicted_cluster].add(truth_cluster)
        truth_predicted_sets[truth_cluster].add(predicted_cluster)
    multi_predicted = [
        cluster for cluster, size in predicted_sizes.items() if size >= 2
    ]
    overmerged = [
        cluster for cluster in multi_predicted if len(predicted_truth_sets[cluster]) > 1
    ]
    overmerged_articles = sum(predicted_sizes[cluster] for cluster in overmerged)
    split_truth = sum(len(rows) > 1 for rows in truth_predicted_sets.values())
    return ClusterMetrics(
        article_count=article_count,
        predicted_cluster_count=len(predicted_sizes),
        pairwise_precision=round(precision, 6),
        pairwise_recall=round(recall, 6),
        pairwise_f1=round(pair_f1, 6),
        b_cubed_precision=round(b_precision, 6),
        b_cubed_recall=round(b_recall, 6),
        b_cubed_f1=round(_harmonic(b_precision, b_recall), 6),
        overmerged_cluster_rate=round(
            len(overmerged) / max(1, len(multi_predicted)), 6
        ),
        overmerged_article_rate=round(overmerged_articles / max(1, article_count), 6),
        split_truth_cluster_rate=round(split_truth / max(1, len(truth_sizes)), 6),
    )


def estimate_cluster_workload(
    articles: list[ArticleRecord], *, time_window_days: int, top_k: int
) -> ClusterWorkloadReport:
    resolved = [article for article in articles if _subject_key(article)]
    by_subject: dict[str, list[ArticleRecord]] = defaultdict(list)
    missing_time = 0
    for article in resolved:
        if article.publish_time is None:
            missing_time += 1
            continue
        by_subject[_subject_key(article)].append(article)
    blocked_pairs = 0
    for rows in by_subject.values():
        rows.sort(key=lambda row: row.publish_time or datetime.min)
        left = 0
        for right, article in enumerate(rows):
            while (
                left < right
                and article.publish_time
                and rows[left].publish_time
                and (article.publish_time - rows[left].publish_time).total_seconds()
                > time_window_days * 86400
            ):
                left += 1
            blocked_pairs += right - left
    article_count = len(articles)
    naive_pairs = _choose_two(article_count)
    reduction = 1.0 - blocked_pairs / naive_pairs if naive_pairs else 0.0
    # 当前 SQLite JSON 向量约按 float32 BLOB 的 4.5 倍估算。
    float32_gb = 300_000 * 1024 * 4 / 1024**3
    return ClusterWorkloadReport(
        article_count=article_count,
        subject_resolved_count=len(resolved),
        naive_pair_count=naive_pairs,
        blocked_pair_count=blocked_pairs,
        reduction_ratio=round(reduction, 6),
        candidate_cap=min(blocked_pairs, article_count * max(1, top_k)),
        unique_subject_count=len(by_subject),
        missing_time_count=missing_time,
        estimated_float32_cache_gb_at_300k=round(float32_gb, 3),
        estimated_json_cache_gb_at_300k=round(float32_gb * 4.5, 3),
        requires_candidate_blocking_before_full_scale=(naive_pairs > 10_000_000),
    )


def _prepare_features(
    dataset: ClusterBenchmarkDataset,
    *,
    cluster_config: dict,
    semantic_scorer: SemanticPairScorer,
    minimum_candidate_threshold: float,
    maximum_top_k: int,
) -> tuple[list[ClusterPairFeature], int, int]:
    article_by_id = {article.article_id: article for article in dataset.articles}
    prediction_by_id = {row.article_id: row for row in dataset.predictions}
    retained: list[ClusterPairFeature] = []
    semantic_candidates: list[ClusterPairFeature] = []
    total_pairs = 0
    for left_article, right_article in itertools.combinations(dataset.articles, 2):
        total_pairs += 1
        left_prediction = prediction_by_id[left_article.article_id]
        right_prediction = prediction_by_id[right_article.article_id]
        rule_score = multi_factor_similarity(
            left_article,
            left_prediction,
            right_article,
            right_prediction,
            cluster_config,
        )
        if rule_score < min(minimum_candidate_threshold, cluster_config["threshold"]):
            continue
        feature = ClusterPairFeature(
            pair_id=_pair_id(left_article.article_id, right_article.article_id),
            left_article_id=left_article.article_id,
            right_article_id=right_article.article_id,
            rule_score=round(rule_score, 6),
        )
        retained.append(feature)
        if (
            rule_score >= minimum_candidate_threshold
            and _subject_key(left_article) == _subject_key(right_article)
            and _within_window(
                left_article.publish_time,
                right_article.publish_time,
                cluster_config["time_window_days"],
            )
        ):
            semantic_candidates.append(feature)

    semantic_selected = _select_top_k(
        semantic_candidates,
        threshold=minimum_candidate_threshold,
        top_k=maximum_top_k,
        require_semantic_score=False,
    )
    semantic_config = cluster_config["semantic"]
    pairs = [
        (
            feature.pair_id,
            _semantic_text(
                article_by_id[feature.left_article_id],
                prediction_by_id[feature.left_article_id],
                semantic_config["max_text_chars"],
            ),
            _semantic_text(
                article_by_id[feature.right_article_id],
                prediction_by_id[feature.right_article_id],
                semantic_config["max_text_chars"],
            ),
        )
        for feature in semantic_selected
    ]
    semantic_scores = semantic_scorer.score_pairs(pairs) if pairs else {}
    by_pair = {feature.pair_id: feature for feature in retained}
    for pair_id, score in semantic_scores.items():
        by_pair[pair_id].semantic_score = score
    return retained, total_pairs, len(semantic_scores)


def _cluster_mapping(
    dataset: ClusterBenchmarkDataset,
    features: list[ClusterPairFeature],
    *,
    rule_threshold: float,
) -> dict[str, str]:
    union_find = _UnionFind([article.article_id for article in dataset.articles])
    for feature in features:
        if feature.rule_score >= rule_threshold:
            union_find.union(feature.left_article_id, feature.right_article_id)
    return {
        article.article_id: union_find.find(article.article_id)
        for article in dataset.articles
    }


def _semantic_cluster_mapping(
    dataset: ClusterBenchmarkDataset,
    features: list[ClusterPairFeature],
    *,
    rule_threshold: float,
    candidate_threshold: float,
    semantic_threshold: float,
    top_k: int,
) -> tuple[dict[str, str], int, int]:
    union_find = _UnionFind([article.article_id for article in dataset.articles])
    for feature in features:
        if feature.rule_score >= rule_threshold:
            union_find.union(feature.left_article_id, feature.right_article_id)
    candidates = _select_top_k(features, threshold=candidate_threshold, top_k=top_k)
    semantic_merged = 0
    for feature in candidates:
        if feature.semantic_score is None:
            continue
        if feature.semantic_score >= semantic_threshold:
            semantic_merged += 1
            union_find.union(feature.left_article_id, feature.right_article_id)
    return (
        {
            article.article_id: union_find.find(article.article_id)
            for article in dataset.articles
        },
        sum(feature.semantic_score is not None for feature in candidates),
        semantic_merged,
    )


def _select_top_k(
    features: list[ClusterPairFeature],
    *,
    threshold: float,
    top_k: int,
    require_semantic_score: bool = True,
) -> list[ClusterPairFeature]:
    by_left: dict[str, list[ClusterPairFeature]] = defaultdict(list)
    for feature in features:
        if (
            feature.rule_score >= threshold
            and (feature.semantic_score is not None or not require_semantic_score)
        ):
            by_left[feature.left_article_id].append(feature)
    selected: dict[str, ClusterPairFeature] = {}
    for rows in by_left.values():
        rows.sort(key=lambda row: (-row.rule_score, row.pair_id))
        for row in rows[: max(1, top_k)]:
            selected[row.pair_id] = row
    return [selected[pair_id] for pair_id in sorted(selected)]


def _semantic_text(
    article: ArticleRecord, prediction: EventPrediction, max_chars: int
) -> str:
    return (
        f"标题：{clean_text(article.title)}\n"
        f"证据：{clean_text(prediction.evidence_sentence)}\n"
        f"正文：{clean_text(article.content)}"
    )[:max_chars]


def _subject_key(article: ArticleRecord) -> str:
    if article.task_scope.startswith("industry") or article.industry_code:
        return article.industry_code or article.industry
    return article.trading_code or article.entity


def _within_window(
    left: datetime | None, right: datetime | None, window_days: int
) -> bool:
    if left is None or right is None:
        return False
    return abs((left - right).total_seconds()) <= window_days * 86400


def _pair_id(left: str, right: str) -> str:
    first, second = sorted((left, right))
    return f"{first}::{second}"


def _choose_two(value: int) -> int:
    return value * (value - 1) // 2


def _harmonic(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left + right else 0.0

