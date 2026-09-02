from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path


from eventlens.config import load_settings
from eventlens.preprocess import clean_text, text_tokens
from eventlens.schema import ArticleRecord, ClusterDecision, EventCluster, EventPrediction
from eventlens.semantic_similarity import SemanticPairScorer


@dataclass
class UnionFind:
    parent: dict[str, str]

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def load_cluster_config(path: str | Path = "configs/app.yaml") -> dict:
    return load_settings(path).cluster.model_dump()


def cluster_events(
    articles: list[ArticleRecord],
    predictions: list[EventPrediction],
    config: dict | None = None,
    semantic_scorer: SemanticPairScorer | None = None,
    decision_sink: list[ClusterDecision] | None = None,
) -> list[EventCluster]:
    cfg = config or load_cluster_config()
    semantic_cfg = cfg.get("semantic", {"enabled": False})
    article_map = {article.article_id: article for article in articles}
    active = [pred for pred in predictions if pred.has_event]
    uf = UnionFind(parent={pred.article_id: pred.article_id for pred in active})
    candidates: list[tuple[str, EventPrediction, EventPrediction, float]] = []

    for i, left in enumerate(active):
        for right in active[i + 1 :]:
            left_article = article_map[left.article_id]
            right_article = article_map[right.article_id]
            score = multi_factor_similarity(left_article, left, right_article, right, cfg)
            if score >= cfg["threshold"]:
                uf.union(left.article_id, right.article_id)
                _record_decision(
                    decision_sink,
                    left,
                    right,
                    rule_score=score,
                    semantic_score=None,
                    merged=True,
                    method="rule",
                    reason="rule_threshold_met",
                )
            elif (
                semantic_cfg.get("enabled", False)
                and semantic_scorer is not None
                and score >= semantic_cfg["candidate_threshold"]
                and _same_subject(left_article, right_article)
                and _within_time_window(
                    left_article.publish_time,
                    right_article.publish_time,
                    cfg["time_window_days"],
                )
            ):
                pair_id = _pair_id(left.article_id, right.article_id)
                candidates.append((pair_id, left, right, score))

    selected = _top_k_candidates(candidates, cfg.get("top_k", 20))
    if selected:
        semantic_pairs = [
            (
                pair_id,
                _semantic_text(
                    article_map[left.article_id],
                    left,
                    semantic_cfg["max_text_chars"],
                ),
                _semantic_text(
                    article_map[right.article_id],
                    right,
                    semantic_cfg["max_text_chars"],
                ),
            )
            for pair_id, left, right, _ in selected
        ]
        try:
            semantic_scores = semantic_scorer.score_pairs(semantic_pairs)
        except Exception as exc:
            if not semantic_cfg.get("fail_open", True):
                raise
            for _, left, right, rule_score in selected:
                _record_decision(
                    decision_sink,
                    left,
                    right,
                    rule_score=rule_score,
                    semantic_score=None,
                    merged=False,
                    method="semantic_fallback",
                    reason=f"semantic_unavailable:{type(exc).__name__}",
                )
        else:
            for pair_id, left, right, rule_score in selected:
                semantic_score = semantic_scores[pair_id]
                merged = semantic_score >= semantic_cfg["similarity_threshold"]
                if merged:
                    uf.union(left.article_id, right.article_id)
                _record_decision(
                    decision_sink,
                    left,
                    right,
                    rule_score=rule_score,
                    semantic_score=semantic_score,
                    merged=merged,
                    method="semantic",
                    reason=(
                        "semantic_threshold_met"
                        if merged
                        else "semantic_threshold_not_met"
                    ),
                )

    groups: dict[str, list[EventPrediction]] = {}
    for pred in active:
        groups.setdefault(uf.find(pred.article_id), []).append(pred)

    clusters: list[EventCluster] = []
    for idx, members in enumerate(groups.values(), start=1):
        member_articles = [article_map[p.article_id] for p in members]
        representative = max(members, key=lambda p: p.classifier_confidence)
        dates = [article.publish_time for article in member_articles if article.publish_time]
        company = next((article.entity for article in member_articles if article.entity), "")
        event_type = representative.event_type
        clusters.append(
            EventCluster(
                event_cluster_id=f"EC-{idx:05d}",
                main_company=company,
                event_type=event_type,
                start_time=min(dates) if dates else None,
                latest_time=max(dates) if dates else None,
                article_ids=[p.article_id for p in members],
                representative_article_id=representative.article_id,
                representative_evidence=representative.evidence_sentence,
                cluster_confidence=sum(p.classifier_confidence for p in members) / len(members),
            )
        )
    return clusters


def _top_k_candidates(
    candidates: list[tuple[str, EventPrediction, EventPrediction, float]],
    top_k: int,
) -> list[tuple[str, EventPrediction, EventPrediction, float]]:
    by_left: dict[str, list[tuple[str, EventPrediction, EventPrediction, float]]] = {}
    for candidate in candidates:
        by_left.setdefault(candidate[1].article_id, []).append(candidate)
    selected: list[tuple[str, EventPrediction, EventPrediction, float]] = []
    for rows in by_left.values():
        rows.sort(key=lambda row: (-row[3], row[0]))
        selected.extend(rows[: max(1, top_k)])
    selected.sort(key=lambda row: row[0])
    return selected


def _same_subject(left: ArticleRecord, right: ArticleRecord) -> bool:
    left_subject = _subject_key(left)
    return bool(left_subject and left_subject == _subject_key(right))


def _subject_key(article: ArticleRecord) -> str:
    if article.task_scope.startswith("industry") or article.industry_code:
        return article.industry_code or article.industry
    return article.trading_code or article.entity


def _within_time_window(
    left: datetime | None,
    right: datetime | None,
    window_days: int,
) -> bool:
    if left is None or right is None:
        return False
    return abs((left - right).total_seconds()) <= window_days * 86400


def _semantic_text(
    article: ArticleRecord,
    prediction: EventPrediction,
    max_chars: int,
) -> str:
    text = (
        f"标题：{clean_text(article.title)}\n"
        f"证据：{clean_text(prediction.evidence_sentence)}\n"
        f"正文：{clean_text(article.content)}"
    )
    return text[:max_chars]


def _pair_id(left_article_id: str, right_article_id: str) -> str:
    left, right = sorted((left_article_id, right_article_id))
    return f"{left}::{right}"


def _record_decision(
    sink: list[ClusterDecision] | None,
    left: EventPrediction,
    right: EventPrediction,
    *,
    rule_score: float,
    semantic_score: float | None,
    merged: bool,
    method: str,
    reason: str,
) -> None:
    if sink is None:
        return
    sink.append(
        ClusterDecision(
            left_article_id=left.article_id,
            right_article_id=right.article_id,
            rule_score=round(rule_score, 6),
            semantic_score=(
                round(semantic_score, 6) if semantic_score is not None else None
            ),
            merged=merged,
            method=method,
            reason=reason,
        )
    )


def multi_factor_similarity(
    left_article: ArticleRecord,
    left_pred: EventPrediction,
    right_article: ArticleRecord,
    right_pred: EventPrediction,
    cfg: dict,
) -> float:
    weights = cfg["weights"]
    left_text = f"{left_article.title} {left_pred.evidence_sentence}"
    right_text = f"{right_article.title} {right_pred.evidence_sentence}"
    text_sim = SequenceMatcher(None, clean_text(left_text), clean_text(right_text)).ratio()
    entity_match = 1.0 if left_article.entity and left_article.entity == right_article.entity else 0.0
    event_type_match = 1.0 if left_pred.event_type == right_pred.event_type else 0.0
    time_close = _time_close(left_article.publish_time, right_article.publish_time, cfg["time_window_days"])
    overlap = _token_overlap(left_text, right_text)
    return (
        weights["text_similarity"] * text_sim
        + weights["entity_match"] * entity_match
        + weights["event_type_match"] * event_type_match
        + weights["time_close"] * time_close
        + weights["key_token_overlap"] * overlap
    )


def _time_close(left: datetime | None, right: datetime | None, window_days: int) -> float:
    if not left or not right:
        return 0.5
    delta_days = abs((left - right).total_seconds()) / 86400
    return max(0.0, 1.0 - min(delta_days, window_days) / window_days)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

