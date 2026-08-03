from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path


from eventlens.config import load_settings
from eventlens.preprocess import clean_text, text_tokens
from eventlens.schema import ArticleRecord, EventCluster, EventPrediction


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
) -> list[EventCluster]:
    cfg = config or load_cluster_config()
    article_map = {article.article_id: article for article in articles}
    active = [pred for pred in predictions if pred.has_event]
    uf = UnionFind(parent={pred.article_id: pred.article_id for pred in active})

    for i, left in enumerate(active):
        for right in active[i + 1 :]:
            score = multi_factor_similarity(article_map[left.article_id], left, article_map[right.article_id], right, cfg)
            if score >= cfg["threshold"]:
                uf.union(left.article_id, right.article_id)

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

