from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from eventlens.candidate_edges import SubjectTimeCandidateEdge
from eventlens.cluster import multi_factor_similarity
from eventlens.cluster_benchmark import ClusterMetrics, calculate_cluster_metrics
from eventlens.event_retrieval import RoutedArticleRecallResult
from eventlens.schema import ArticleRecord, ClusterDecision, EventCluster, EventPrediction


class CandidateClusterDecision(BaseModel):
    left_article_id: str
    right_article_id: str
    semantic_score: float = Field(ge=-1.0, le=1.0)
    event_consistent: bool
    merged: bool
    reason: str


class CandidateClusterReport(BaseModel):
    article_count: int
    edge_count: int
    event_consistent_edge_count: int
    semantic_pass_edge_count: int
    merged_edge_count: int
    similarity_threshold: float
    metrics: ClusterMetrics


class CandidateRuleBaselineReport(BaseModel):
    article_count: int
    edge_count: int
    merged_edge_count: int
    rule_threshold: float
    metrics: ClusterMetrics


@dataclass
class _UnionFind:
    parent: dict[str, str]

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_candidate_events(
    articles: list[ArticleRecord],
    predictions: list[EventPrediction],
    article_ids: list[str],
    vectors,
    edges: list[SubjectTimeCandidateEdge],
    recalls: list[RoutedArticleRecallResult],
    *,
    similarity_threshold: float,
) -> tuple[list[EventCluster], list[ClusterDecision]]:
    """生产聚类：候选边上仅合并事件 Top-1 一致且语义相似的有效事件。"""

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold 必须位于 [-1, 1]")
    if not (
        len(articles)
        == len(predictions)
        == len(article_ids)
        == len(vectors)
        == len(recalls)
    ):
        raise ValueError("articles、predictions、article_ids、vectors、recalls 数量不一致")
    if article_ids != [row.article_id for row in articles]:
        raise ValueError("文章顺序与 embedding index 不一致")
    if article_ids != [row.article_id for row in predictions]:
        raise ValueError("预测顺序与 embedding index 不一致")
    if article_ids != [row.article_id for row in recalls]:
        raise ValueError("事件召回顺序与 embedding index 不一致")

    article_by_id = {row.article_id: row for row in articles}
    prediction_by_id = {row.article_id: row for row in predictions}
    recall_by_id = {row.article_id: row for row in recalls}
    active_ids = {row.article_id for row in predictions if row.has_event}
    uf = _UnionFind(parent={article_id: article_id for article_id in active_ids})
    index_by_id = {article_id: idx for idx, article_id in enumerate(article_ids)}
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    decisions: list[ClusterDecision] = []

    for edge in edges:
        if edge.left_article_id not in index_by_id or edge.right_article_id not in index_by_id:
            raise ValueError("候选边包含当前 embedding 之外的 article_id")
        if edge.left_article_id not in active_ids or edge.right_article_id not in active_ids:
            continue
        left_recall = recall_by_id[edge.left_article_id]
        right_recall = recall_by_id[edge.right_article_id]
        event_consistent = (
            _top1_event_key(left_recall) is not None
            and _top1_event_key(left_recall) == _top1_event_key(right_recall)
        )
        left_idx = index_by_id[edge.left_article_id]
        right_idx = index_by_id[edge.right_article_id]
        denominator = float(norms[left_idx] * norms[right_idx])
        semantic_score = (
            float(np.dot(matrix[left_idx], matrix[right_idx]) / denominator)
            if denominator > 0.0
            else 0.0
        )
        semantic_pass = semantic_score >= similarity_threshold
        merged = event_consistent and semantic_pass
        if merged:
            uf.union(edge.left_article_id, edge.right_article_id)
        decisions.append(
            ClusterDecision(
                left_article_id=edge.left_article_id,
                right_article_id=edge.right_article_id,
                rule_score=0.0,
                semantic_score=round(semantic_score, 6),
                merged=merged,
                method="candidate_semantic",
                reason=(
                    "event_and_semantic_pass"
                    if merged
                    else "event_top1_mismatch"
                    if not event_consistent
                    else "semantic_threshold_not_met"
                ),
            )
        )

    groups: dict[str, list[str]] = {}
    for article_id in article_ids:
        if article_id in active_ids:
            groups.setdefault(uf.find(article_id), []).append(article_id)

    clusters: list[EventCluster] = []
    for member_ids in groups.values():
        member_predictions = [prediction_by_id[article_id] for article_id in member_ids]
        member_articles = [article_by_id[article_id] for article_id in member_ids]
        representative = max(
            member_predictions,
            key=lambda row: (row.classifier_confidence, row.article_id),
        )
        representative_recall = recall_by_id[representative.article_id]
        top_recall = representative_recall.candidates[0] if representative_recall.candidates else None
        subject = representative.event_subject or (top_recall.subject_name if top_recall else "")
        dates = [row.publish_time for row in member_articles if row.publish_time]
        cluster_key = "|".join(sorted(member_ids))
        cluster_id = f"EC-{hashlib.sha256(cluster_key.encode('utf-8')).hexdigest()[:12]}"
        clusters.append(
            EventCluster(
                event_cluster_id=cluster_id,
                main_company=subject,
                event_type=representative.event_type,
                start_time=min(dates) if dates else None,
                latest_time=max(dates) if dates else None,
                article_ids=sorted(member_ids),
                representative_article_id=representative.article_id,
                representative_evidence=representative.evidence_sentence,
                cluster_confidence=round(
                    sum(row.classifier_confidence for row in member_predictions)
                    / len(member_predictions),
                    6,
                ),
            )
        )
    clusters.sort(key=lambda row: row.event_cluster_id)
    return clusters, decisions


def evaluate_candidate_clusters(
    article_ids: list[str],
    vectors,
    edges: list[SubjectTimeCandidateEdge],
    recalls: list[RoutedArticleRecallResult],
    truth_by_article: dict[str, str],
    *,
    similarity_threshold: float,
) -> tuple[CandidateClusterReport, list[CandidateClusterDecision]]:
    """仅在事件 Top-1 一致且文章向量相似度过门槛时合并候选边。"""

    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold 必须位于 [-1, 1]")
    if len(article_ids) != len(vectors) or len(article_ids) != len(recalls):
        raise ValueError("article_ids、vectors 与 recalls 数量不一致")
    recall_ids = [row.article_id for row in recalls]
    if article_ids != recall_ids:
        raise ValueError("事件召回顺序与 embedding index 不一致")
    if set(truth_by_article) != set(article_ids):
        raise ValueError("truth_by_article 必须覆盖且仅覆盖当前文章")

    index_by_id = {article_id: idx for idx, article_id in enumerate(article_ids)}
    event_key_by_id = {
        row.article_id: _top1_event_key(row)
        for row in recalls
    }
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    uf = _UnionFind(parent={article_id: article_id for article_id in article_ids})
    decisions: list[CandidateClusterDecision] = []
    event_consistent_count = 0
    semantic_pass_count = 0
    merged_count = 0

    for edge in edges:
        if edge.left_article_id not in index_by_id or edge.right_article_id not in index_by_id:
            raise ValueError("候选边包含当前 embedding 之外的 article_id")
        left_idx = index_by_id[edge.left_article_id]
        right_idx = index_by_id[edge.right_article_id]
        denominator = float(norms[left_idx] * norms[right_idx])
        score = (
            float(np.dot(matrix[left_idx], matrix[right_idx]) / denominator)
            if denominator > 0.0
            else 0.0
        )
        left_event = event_key_by_id[edge.left_article_id]
        right_event = event_key_by_id[edge.right_article_id]
        event_consistent = left_event is not None and left_event == right_event
        semantic_pass = score >= similarity_threshold
        merged = event_consistent and semantic_pass
        event_consistent_count += int(event_consistent)
        semantic_pass_count += int(semantic_pass)
        merged_count += int(merged)
        if merged:
            uf.union(edge.left_article_id, edge.right_article_id)
        decisions.append(
            CandidateClusterDecision(
                left_article_id=edge.left_article_id,
                right_article_id=edge.right_article_id,
                semantic_score=round(score, 6),
                event_consistent=event_consistent,
                merged=merged,
                reason=(
                    "event_and_semantic_pass"
                    if merged
                    else "event_top1_mismatch"
                    if not event_consistent
                    else "semantic_threshold_not_met"
                ),
            )
        )

    predicted = {article_id: uf.find(article_id) for article_id in article_ids}
    report = CandidateClusterReport(
        article_count=len(article_ids),
        edge_count=len(edges),
        event_consistent_edge_count=event_consistent_count,
        semantic_pass_edge_count=semantic_pass_count,
        merged_edge_count=merged_count,
        similarity_threshold=similarity_threshold,
        metrics=calculate_cluster_metrics(truth_by_article, predicted),
    )
    return report, decisions


def evaluate_candidate_rule_baseline(
    articles: list[ArticleRecord],
    edges: list[SubjectTimeCandidateEdge],
    recalls: list[RoutedArticleRecallResult],
    truth_by_article: dict[str, str],
    *,
    cluster_config: dict,
) -> CandidateRuleBaselineReport:
    """在完全相同候选边上执行现有多因子规则，作为发布门禁对照。"""

    article_ids = [row.article_id for row in articles]
    recall_ids = [row.article_id for row in recalls]
    if article_ids != recall_ids:
        raise ValueError("事件召回顺序与评测文章不一致")
    if set(truth_by_article) != set(article_ids):
        raise ValueError("truth_by_article 必须覆盖且仅覆盖当前文章")

    articles_by_id: dict[str, ArticleRecord] = {}
    predictions_by_id: dict[str, EventPrediction] = {}
    for article, recall in zip(articles, recalls, strict=True):
        top = recall.candidates[0] if recall.candidates else None
        subject_code = top.subject_code if top else ""
        event_name = top.event_name if top else ""
        # 候选层已经完成主体约束，重复新闻原表缺主体字段时使用预测主体。
        articles_by_id[article.article_id] = article.model_copy(
            update={"entity": subject_code}
        )
        evidence = (article.content or article.title)[:160]
        predictions_by_id[article.article_id] = EventPrediction(
            article_id=article.article_id,
            has_event=bool(top),
            event_type=event_name,
            classifier_confidence=float(top.score) if top else 0.0,
            event_subject=subject_code or None,
            event_time=article.publish_time,
            evidence_sentence=evidence,
            extraction_confidence=float(top.score) if top else 0.0,
        )

    threshold = float(cluster_config["threshold"])
    uf = _UnionFind(parent={article_id: article_id for article_id in article_ids})
    merged_count = 0
    for edge in edges:
        if edge.left_article_id not in articles_by_id or edge.right_article_id not in articles_by_id:
            raise ValueError("候选边包含当前评测文章之外的 article_id")
        left_article = articles_by_id[edge.left_article_id]
        right_article = articles_by_id[edge.right_article_id]
        score = multi_factor_similarity(
            left_article,
            predictions_by_id[edge.left_article_id],
            right_article,
            predictions_by_id[edge.right_article_id],
            cluster_config,
        )
        if score >= threshold:
            uf.union(edge.left_article_id, edge.right_article_id)
            merged_count += 1

    predicted = {article_id: uf.find(article_id) for article_id in article_ids}
    return CandidateRuleBaselineReport(
        article_count=len(article_ids),
        edge_count=len(edges),
        merged_edge_count=merged_count,
        rule_threshold=threshold,
        metrics=calculate_cluster_metrics(truth_by_article, predicted),
    )


def _top1_event_key(row: RoutedArticleRecallResult) -> tuple[str, str] | None:
    if not row.candidates:
        return None
    top = row.candidates[0]
    return top.subject_code, top.event_name
