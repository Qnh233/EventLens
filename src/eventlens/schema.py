from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ArticleRecord(BaseModel):
    article_id: str
    title: str = ""
    publish_time: datetime | None = None
    source: str = ""
    content: str = ""
    entity: str = ""
    industry: str = ""
    event_label: str | None = None
    polarity_label: str | None = None
    impact_analysis: str | None = None
    duplicate_flag: str | int | bool | None = None
    source_row: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EventPrediction(BaseModel):
    article_id: str
    has_event: bool
    event_type: str
    event_polarity: str = "中性"
    classifier_confidence: float = 0.0
    polarity_confidence: float = 0.0
    event_subject: str | None = None
    event_time: datetime | None = None
    impact_target: str | None = None
    impact_direction: str = "中性"
    impact_description: str = ""
    evidence_sentence: str = ""
    extraction_confidence: float = 0.0


class EventCluster(BaseModel):
    event_cluster_id: str
    main_company: str = ""
    event_type: str
    start_time: datetime | None = None
    latest_time: datetime | None = None
    article_ids: list[str]
    representative_article_id: str
    representative_evidence: str = ""
    cluster_confidence: float = 0.0


class CredibilityBreakdown(BaseModel):
    source_authority: float
    multi_source_consistency: float
    official_endorsement: float
    content_completeness: float
    recency_score: float
    duplicate_penalty: float = 0.0
    conflict_penalty: float = 0.0


class AlertOutput(BaseModel):
    alert_id: str
    risk_level: str
    impact_direction: str = "中性"
    event_cluster_id: str
    company: str = ""
    event_type: str
    event_summary: str
    credibility_score: float
    severity_score: float
    credibility_breakdown: CredibilityBreakdown
    evidence_sources: list[dict[str, str]]
    push_reason: str
    related_article_count: int
    is_throttled_update: bool = False
    dormant: bool = False

