from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import heapq
import json
from pathlib import Path

from pydantic import BaseModel, Field

from eventlens.schema import ArticleRecord
from eventlens.subject_routing import SubjectRouteResult


class SubjectTimeCandidateEdge(BaseModel):
    """主体候选与时间窗共同约束的可审计文章对。"""

    left_article_id: str
    right_article_id: str
    shared_subject_codes: list[str]
    subject_score: float = Field(ge=-1.0, le=1.0)
    day_gap: float = Field(ge=0.0)
    left_hard_routed: bool
    right_hard_routed: bool


class CandidateEdgeRecallReport(BaseModel):
    positive_pair_count: int
    eligible_positive_pair_count: int
    retrieved_positive_pair_count: int
    retrieved_eligible_positive_pair_count: int
    positive_pair_recall: float = Field(ge=0.0, le=1.0)
    eligible_positive_pair_recall: float = Field(ge=0.0, le=1.0)
    edge_count: int
    minimum_eligible_recall: float = Field(ge=0.0, le=1.0)
    passed: bool


def build_subject_time_edges(
    articles: list[ArticleRecord],
    routes: list[SubjectRouteResult],
    *,
    window_days: int = 7,
    max_neighbors_per_article: int = 20,
) -> list[SubjectTimeCandidateEdge]:
    """用主体倒排桶 + 时间窗 + Top-K 生成稀疏候选边。"""

    if window_days < 0:
        raise ValueError("window_days 不能小于 0")
    if max_neighbors_per_article < 1:
        raise ValueError("max_neighbors_per_article 必须大于 0")
    if len(articles) != len(routes):
        raise ValueError("文章数量与主体路由数量不一致")
    article_ids = [article.article_id for article in articles]
    route_ids = [route.article_id for route in routes]
    if article_ids != route_ids:
        raise ValueError("主体路由 article_id 顺序与文章顺序不一致")

    article_by_id = {article.article_id: article for article in articles}
    route_by_id = {route.article_id: route for route in routes}
    buckets: dict[str, list[tuple[datetime, str, float]]] = defaultdict(list)
    for article, route in zip(articles, routes):
        if article.publish_time is None:
            continue
        for code, score in _route_subject_scores(route).items():
            buckets[code].append((article.publish_time, article.article_id, score))

    # 每篇文章最终只保留固定数量的历史近邻，避免宽行业 Top-K 在 7 天窗口内
    # 产生近似平方级边数。候选优先级先看双方保守主体分数，再看时间接近度。
    selected_by_right: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    max_seconds = window_days * 86400
    for code, rows in buckets.items():
        rows.sort(key=lambda row: (row[0], row[1]))
        # 最大堆按主体候选分数取窗口内最可信的历史文章；过期项惰性清理，
        # 每个过期项最多被真正丢弃一次，避免逐文章扫描整个 7 天窗口。
        score_heap: list[tuple[float, datetime, str]] = []
        for right_time, right_id, right_score in rows:
            picked: list[tuple[float, datetime, str]] = []
            valid: list[tuple[float, datetime, str]] = []
            while score_heap and len(picked) < max_neighbors_per_article:
                item = heapq.heappop(score_heap)
                _, left_time, _ = item
                if (right_time - left_time).total_seconds() > max_seconds:
                    continue
                picked.append(item)
                valid.append(item)
            for item in valid:
                heapq.heappush(score_heap, item)

            for negative_left_score, left_time, left_id in picked:
                left_score = -negative_left_score
                subject_score = min(left_score, right_score)
                day_gap = (right_time - left_time).total_seconds() / 86400
                _keep_bounded_neighbor(
                    selected_by_right[right_id],
                    left_id=left_id,
                    subject_score=subject_score,
                    day_gap=day_gap,
                    limit=max_neighbors_per_article,
                )
            heapq.heappush(score_heap, (-right_score, right_time, right_id))

    output: list[SubjectTimeCandidateEdge] = []
    route_scores = {
        route.article_id: _route_subject_scores(route) for route in routes
    }
    pairs = sorted(
        (left_id, right_id)
        for right_id, neighbors in selected_by_right.items()
        for left_id in neighbors
    )
    for left_id, right_id in pairs:
        left_article = article_by_id[left_id]
        right_article = article_by_id[right_id]
        if left_article.publish_time is None or right_article.publish_time is None:
            continue
        shared_codes = sorted(
            set(route_scores[left_id]).intersection(route_scores[right_id])
        )
        if not shared_codes:
            continue
        shared_scores = {
            code: min(route_scores[left_id][code], route_scores[right_id][code])
            for code in shared_codes
        }
        day_gap = abs(
            (left_article.publish_time - right_article.publish_time).total_seconds()
        ) / 86400
        left_route = route_by_id[left_id]
        right_route = route_by_id[right_id]
        output.append(
            SubjectTimeCandidateEdge(
                left_article_id=left_id,
                right_article_id=right_id,
                shared_subject_codes=shared_codes,
                subject_score=round(max(shared_scores.values()), 6),
                day_gap=round(day_gap, 6),
                left_hard_routed=left_route.accepted_subject_code is not None,
                right_hard_routed=right_route.accepted_subject_code is not None,
            )
        )
    return output


def evaluate_duplicate_candidate_recall(
    articles: list[ArticleRecord],
    edges: list[SubjectTimeCandidateEdge],
    *,
    window_days: int = 7,
    minimum_eligible_recall: float = 0.95,
) -> CandidateEdgeRecallReport:
    """用 duplication_id 检查候选层是否漏掉时间窗内应召回的同源正对。"""

    if window_days < 0:
        raise ValueError("window_days 不能小于 0")
    if not 0.0 <= minimum_eligible_recall <= 1.0:
        raise ValueError("minimum_eligible_recall 必须位于 [0, 1]")

    groups: dict[str, list[ArticleRecord]] = defaultdict(list)
    for article in articles:
        if article.duplication_id:
            groups[article.duplication_id].append(article)

    positive_pairs: set[tuple[str, str]] = set()
    eligible_pairs: set[tuple[str, str]] = set()
    for rows in groups.values():
        ordered = sorted(rows, key=lambda row: row.article_id)
        for idx, left in enumerate(ordered):
            for right in ordered[idx + 1 :]:
                pair = _normalized_pair(left.article_id, right.article_id)
                positive_pairs.add(pair)
                if left.publish_time is None or right.publish_time is None:
                    continue
                day_gap = abs(
                    (left.publish_time - right.publish_time).total_seconds()
                ) / 86400
                if day_gap <= window_days:
                    eligible_pairs.add(pair)

    edge_pairs = {
        _normalized_pair(edge.left_article_id, edge.right_article_id) for edge in edges
    }
    retrieved_positive = positive_pairs.intersection(edge_pairs)
    retrieved_eligible = eligible_pairs.intersection(edge_pairs)
    positive_recall = (
        len(retrieved_positive) / len(positive_pairs) if positive_pairs else 1.0
    )
    eligible_recall = (
        len(retrieved_eligible) / len(eligible_pairs) if eligible_pairs else 1.0
    )
    return CandidateEdgeRecallReport(
        positive_pair_count=len(positive_pairs),
        eligible_positive_pair_count=len(eligible_pairs),
        retrieved_positive_pair_count=len(retrieved_positive),
        retrieved_eligible_positive_pair_count=len(retrieved_eligible),
        positive_pair_recall=round(positive_recall, 6),
        eligible_positive_pair_recall=round(eligible_recall, 6),
        edge_count=len(edges),
        minimum_eligible_recall=minimum_eligible_recall,
        passed=eligible_recall >= minimum_eligible_recall,
    )


def load_candidate_edges_jsonl(path: str | Path) -> list[SubjectTimeCandidateEdge]:
    """读取候选边产物，供独立门禁评测复用。"""

    rows: list[SubjectTimeCandidateEdge] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(SubjectTimeCandidateEdge.model_validate(json.loads(line)))
    return rows


def _keep_bounded_neighbor(
    neighbors: dict[str, tuple[float, float]],
    *,
    left_id: str,
    subject_score: float,
    day_gap: float,
    limit: int,
) -> None:
    current = neighbors.get(left_id)
    candidate = (subject_score, day_gap)
    if current is not None:
        if _neighbor_rank(candidate, left_id) > _neighbor_rank(current, left_id):
            neighbors[left_id] = candidate
        return
    if len(neighbors) < limit:
        neighbors[left_id] = candidate
        return
    worst_id, worst = min(
        neighbors.items(), key=lambda item: _neighbor_rank(item[1], item[0])
    )
    if _neighbor_rank(candidate, left_id) > _neighbor_rank(worst, worst_id):
        del neighbors[worst_id]
        neighbors[left_id] = candidate


def _neighbor_rank(value: tuple[float, float], article_id: str) -> tuple[float, float, str]:
    subject_score, day_gap = value
    return subject_score, -day_gap, article_id


def _route_subject_scores(route: SubjectRouteResult) -> dict[str, float]:
    scores = {row.subject_code: row.score for row in route.candidates}
    if route.accepted_subject_code:
        scores[route.accepted_subject_code] = max(
            route.top1_score,
            scores.get(route.accepted_subject_code, -1.0),
        )
    return scores


def _normalized_pair(left_id: str, right_id: str) -> tuple[str, str]:
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)

