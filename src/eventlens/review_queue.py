from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from eventlens.learning import FeedbackRecord
from eventlens.schema import ArticleRecord


class ReviewQueueItem(BaseModel):
    review_id: str
    article_id: str
    scope: str
    reason: str
    priority: float = Field(ge=0.0, le=1.0)
    baseline_event: str
    svc_margin: float
    candidate_events: list[str] = Field(default_factory=list)
    route_status: str
    status: str = "pending"
    requires_human_approval: bool = True
    provenance_hash: str
    created_at: datetime


class HumanReviewDecision(BaseModel):
    review_id: str
    provenance_hash: str
    expected_event_type: str
    reviewer: str
    approved: bool
    note: str = ""


def build_human_review_packet(
    queue: list[ReviewQueueItem],
    articles: list[ArticleRecord],
    *,
    content_chars: int = 1200,
) -> list[dict[str, object]]:
    """生成不含 Gold/主体真值字段的人工审阅上下文。"""
    article_by_id: dict[str, ArticleRecord] = {}
    for row in articles:
        existing = article_by_id.get(row.article_id)
        if existing is None:
            article_by_id[row.article_id] = row
            continue
        # 真实训练集存在重复 article_id；仅当公开审阅上下文完全一致时安全去重。
        current_context = (row.title, row.content, row.source, row.publish_time)
        existing_context = (existing.title, existing.content, existing.source, existing.publish_time)
        if current_context != existing_context:
            raise ValueError(f"conflicting duplicate article_id in review source: {row.article_id}")

    packet: list[dict[str, object]] = []
    for item in queue:
        article = article_by_id.get(item.article_id)
        if article is None:
            raise ValueError(f"review article not found: {item.article_id}")
        packet.append(
            {
                "review_id": item.review_id,
                "provenance_hash": item.provenance_hash,
                "article_id": item.article_id,
                "title": article.title,
                "source": article.source,
                "publish_time": article.publish_time.isoformat() if article.publish_time else None,
                "content_preview": article.content[: max(0, content_chars)],
                "baseline_event": item.baseline_event,
                "candidate_events": item.candidate_events,
                "svc_margin": item.svc_margin,
                "route_status": item.route_status,
                "expected_event_type": "",
                "reviewer": "",
                "approved": False,
                "note": "",
            }
        )
    return packet


def load_review_queue_jsonl(path: str | Path) -> list[ReviewQueueItem]:
    rows: list[ReviewQueueItem] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(ReviewQueueItem.model_validate(json.loads(line)))
    return rows


def load_human_review_jsonl(path: str | Path) -> list[HumanReviewDecision]:
    rows: list[HumanReviewDecision] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(HumanReviewDecision.model_validate(json.loads(line)))
    return rows


def convert_reviews_to_feedback(
    queue: list[ReviewQueueItem],
    reviews: list[HumanReviewDecision],
    *,
    valid_event_types: set[str],
) -> list[FeedbackRecord]:
    """只把通过人工审批且 provenance 完整的复核转成训练反馈。"""
    queue_by_id = {row.review_id: row for row in queue}
    if len(queue_by_id) != len(queue):
        raise ValueError("duplicate review_id in queue")

    feedback: list[FeedbackRecord] = []
    seen_review_ids: set[str] = set()
    for review in reviews:
        if review.review_id in seen_review_ids:
            raise ValueError(f"duplicate reviewed decision: {review.review_id}")
        seen_review_ids.add(review.review_id)
        source = queue_by_id.get(review.review_id)
        if source is None:
            raise ValueError(f"unknown review_id: {review.review_id}")
        if review.provenance_hash != source.provenance_hash:
            raise ValueError(f"provenance mismatch: {review.review_id}")
        if not review.approved:
            continue
        if not review.reviewer.strip():
            raise ValueError(f"reviewer required: {review.review_id}")
        if review.expected_event_type not in valid_event_types:
            raise ValueError(f"unknown event type: {review.expected_event_type}")

        seed = "|".join(
            [
                review.review_id,
                source.provenance_hash,
                review.expected_event_type,
                review.reviewer.strip(),
            ]
        )
        feedback_id = f"FB-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"
        feedback.append(
            FeedbackRecord(
                feedback_id=feedback_id,
                kind="wrong_type" if review.expected_event_type != source.baseline_event else "confirmed_type",
                article_id=source.article_id,
                expected_event_type=review.expected_event_type,
                observed_event_type=source.baseline_event,
                note=review.note,
                created_at=datetime.now(timezone.utc),
                reviewer=review.reviewer.strip(),
                metadata={
                    "review_id": source.review_id,
                    "review_provenance_hash": source.provenance_hash,
                    "candidate_hit": review.expected_event_type in source.candidate_events,
                    "route_status": source.route_status,
                    "svc_margin": source.svc_margin,
                    "requires_human_approval": source.requires_human_approval,
                },
            )
        )
    return feedback


def build_low_margin_review_items(
    *,
    article_ids: list[str],
    scope: str,
    baseline_events: list[str],
    margins: list[float],
    candidate_events: list[list[str]],
    route_statuses: list[str],
    selected_indices: list[int],
    reason: str = "low_margin_top20pct",
) -> list[ReviewQueueItem]:
    size = len(article_ids)
    if not all(
        len(values) == size
        for values in (baseline_events, margins, candidate_events, route_statuses)
    ):
        raise ValueError("review queue input count mismatch")

    rows: list[ReviewQueueItem] = []
    for rank, index in enumerate(selected_indices):
        if not 0 <= index < size:
            raise IndexError("review queue index out of range")
        candidates = list(dict.fromkeys(candidate_events[index]))
        seed = "|".join(
            [
                scope,
                article_ids[index],
                baseline_events[index],
                f"{float(margins[index]):.8f}",
                *candidates,
            ]
        )
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        # 越靠前的低 margin 样本优先级越高，但不声称其一定错误。
        priority = 1.0 - (rank / max(1, len(selected_indices)))
        rows.append(
            ReviewQueueItem(
                review_id=f"RQ-{digest[:16]}",
                article_id=article_ids[index],
                scope=scope,
                reason=reason,
                priority=round(priority, 6),
                baseline_event=baseline_events[index],
                svc_margin=round(float(margins[index]), 6),
                candidate_events=candidates,
                route_status=route_statuses[index],
                provenance_hash=digest,
                created_at=datetime.now(timezone.utc),
            )
        )
    return rows
