from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from eventlens.config import load_settings
from eventlens.credibility import source_authority_score
from eventlens.schema import ArticleRecord, EventCluster, EventPrediction


class LifecycleStage(str, Enum):
    DISCOVERED = "发现"
    MEDIA_REPORTED = "媒体报道"
    CORROBORATED = "多源佐证"
    COMPANY_RESPONDED = "公司回应"
    OFFICIAL_CONFIRMED = "官方确认"
    DISPUTED = "存在争议"
    OFFICIAL_CLARIFIED = "官方澄清"
    IMPACT_ESCALATED = "影响扩大"
    RESOLVED = "已处置"


class ClaimStatus(str, Enum):
    UNVERIFIED = "未核实"
    SUPPORTED = "得到支持"
    CONFIRMED = "已确认"
    DISPUTED = "存在争议"
    REFUTED = "已证伪"
    RESOLVED = "已完结"


class EvidenceStance(str, Enum):
    SUPPORTS = "支持"
    REFUTES = "反驳"
    NEUTRAL = "中性"


class EvidenceRecord(BaseModel):
    evidence_id: str
    article_id: str
    source: str
    source_type: str
    independent_source_key: str
    stance: EvidenceStance
    authority_score: float = Field(ge=0.0, le=1.0)
    published_at: datetime | None = None
    summary: str = ""
    content_hash: str


class CredibilitySnapshot(BaseModel):
    sequence: int
    observed_at: datetime | None = None
    previous_score: float = Field(ge=0.0, le=1.0)
    credibility_score: float = Field(ge=0.0, le=1.0)
    delta: float
    stage: LifecycleStage
    claim_status: ClaimStatus
    evidence_ids: list[str] = Field(default_factory=list)
    change_reason: str


class EventLifecycle(BaseModel):
    event_cluster_id: str
    company: str = ""
    event_type: str
    stage: LifecycleStage = LifecycleStage.DISCOVERED
    claim_status: ClaimStatus = ClaimStatus.UNVERIFIED
    credibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    snapshots: list[CredibilitySnapshot] = Field(default_factory=list)
    version: int = 1
    created_at: datetime
    updated_at: datetime


class EventLifecycleLedger:
    """追加式生命周期账本；旧版本永不覆盖，便于审计和回放。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, lifecycles: list[EventLifecycle]) -> None:
        if not lifecycles:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        recorded_at = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as file:
            for lifecycle in lifecycles:
                payload = lifecycle.model_dump(mode="json")
                payload["ledger_recorded_at"] = recorded_at
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_all(self) -> list[EventLifecycle]:
        if not self.path.exists():
            return []
        rows: list[EventLifecycle] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                payload.pop("ledger_recorded_at", None)
                rows.append(EventLifecycle.model_validate(payload))
        return rows

    def latest(self) -> dict[str, EventLifecycle]:
        latest_rows: dict[str, EventLifecycle] = {}
        for row in self.read_all():
            current = latest_rows.get(row.event_cluster_id)
            if current is None or row.version >= current.version:
                latest_rows[row.event_cluster_id] = row
        return latest_rows


def build_event_lifecycles(
    articles: list[ArticleRecord],
    predictions: list[EventPrediction],
    clusters: list[EventCluster],
    lifecycle_config: dict | None = None,
    credibility_config: dict | None = None,
) -> list[EventLifecycle]:
    settings = load_settings()
    lifecycle_cfg = lifecycle_config or settings.lifecycle.model_dump()
    credibility_cfg = credibility_config or settings.credibility.model_dump()
    article_map = {article.article_id: article for article in articles}
    prediction_map = {prediction.article_id: prediction for prediction in predictions}

    lifecycles: list[EventLifecycle] = []
    for cluster in clusters:
        cluster_articles = [
            article_map[article_id]
            for article_id in cluster.article_ids
            if article_id in article_map
        ]
        cluster_articles.sort(key=lambda row: (row.publish_time is None, row.publish_time or datetime.max))
        lifecycles.append(
            _build_lifecycle(
                cluster,
                cluster_articles,
                prediction_map,
                lifecycle_cfg,
                credibility_cfg,
            )
        )
    return lifecycles


def _build_lifecycle(
    cluster: EventCluster,
    articles: list[ArticleRecord],
    prediction_map: dict[str, EventPrediction],
    lifecycle_cfg: dict,
    credibility_cfg: dict,
) -> EventLifecycle:
    now = datetime.now(timezone.utc)
    created_at = next((article.publish_time for article in articles if article.publish_time), now)
    lifecycle = EventLifecycle(
        event_cluster_id=cluster.event_cluster_id,
        company=cluster.main_company,
        event_type=cluster.event_type,
        created_at=created_at,
        updated_at=created_at,
    )
    source_keys: set[str] = set()

    for sequence, article in enumerate(articles, start=1):
        prediction = prediction_map.get(article.article_id)
        evidence = _to_evidence(article, prediction, lifecycle_cfg, credibility_cfg)
        previous_score = lifecycle.credibility_score
        is_new_source = evidence.independent_source_key not in source_keys
        source_keys.add(evidence.independent_source_key)
        lifecycle.evidence.append(evidence)

        stage, claim_status, reason = _derive_state(lifecycle.evidence, lifecycle_cfg)
        new_score = _evolve_credibility(
            previous_score,
            evidence,
            is_new_source,
            stage,
            lifecycle_cfg,
        )
        lifecycle.stage = stage
        lifecycle.claim_status = claim_status
        lifecycle.credibility_score = new_score
        lifecycle.updated_at = article.publish_time or now
        lifecycle.snapshots.append(
            CredibilitySnapshot(
                sequence=sequence,
                observed_at=article.publish_time,
                previous_score=round(previous_score, 4),
                credibility_score=round(new_score, 4),
                delta=round(new_score - previous_score, 4),
                stage=stage,
                claim_status=claim_status,
                evidence_ids=[row.evidence_id for row in lifecycle.evidence],
                change_reason=reason,
            )
        )

    return lifecycle


def _to_evidence(
    article: ArticleRecord,
    prediction: EventPrediction | None,
    lifecycle_cfg: dict,
    credibility_cfg: dict,
) -> EvidenceRecord:
    text = f"{article.title} {article.content} {prediction.evidence_sentence if prediction else ''}".strip()
    stance = _infer_stance(text, lifecycle_cfg)
    authority = source_authority_score(article.source, credibility_cfg)
    source_type = _infer_source_type(article.source, credibility_cfg)
    content_hash = hashlib.sha256(
        f"{article.source}|{article.title}|{article.content}".encode("utf-8")
    ).hexdigest()
    evidence_id = f"EVD-{content_hash[:16]}"
    summary = prediction.evidence_sentence if prediction and prediction.evidence_sentence else article.title
    return EvidenceRecord(
        evidence_id=evidence_id,
        article_id=article.article_id,
        source=article.source or "未知来源",
        source_type=source_type,
        independent_source_key=_normalize_source_key(article.source, content_hash),
        stance=stance,
        authority_score=authority,
        published_at=article.publish_time,
        summary=summary[:300],
        content_hash=content_hash,
    )


def _infer_stance(text: str, lifecycle_cfg: dict) -> EvidenceStance:
    if any(keyword in text for keyword in lifecycle_cfg["refutation_keywords"]):
        return EvidenceStance.REFUTES
    if any(keyword in text for keyword in lifecycle_cfg["support_keywords"]):
        return EvidenceStance.SUPPORTS
    return EvidenceStance.NEUTRAL


def _infer_source_type(source: str, credibility_cfg: dict) -> str:
    source = source or ""
    keywords = credibility_cfg["source_authority"]
    ordered_types = (
        ("官方", "official_keywords"),
        ("公司", "company_keywords"),
        ("主流财经媒体", "mainstream_keywords"),
        ("综合媒体", "general_keywords"),
        ("自媒体", "self_media_keywords"),
    )
    for source_type, key in ordered_types:
        if any(keyword in source for keyword in keywords[key]):
            return source_type
    return "未知来源"


def _normalize_source_key(source: str, fallback_hash: str) -> str:
    normalized = "".join((source or "").lower().split())
    return normalized or f"unknown-{fallback_hash[:12]}"


def _derive_state(
    evidence: list[EvidenceRecord],
    lifecycle_cfg: dict,
) -> tuple[LifecycleStage, ClaimStatus, str]:
    latest = evidence[-1]
    support_rows = [row for row in evidence if row.stance == EvidenceStance.SUPPORTS]
    refute_rows = [row for row in evidence if row.stance == EvidenceStance.REFUTES]
    independent_support = {row.independent_source_key for row in support_rows}
    official_threshold = lifecycle_cfg["official_authority_threshold"]
    company_threshold = lifecycle_cfg["company_authority_threshold"]
    latest_text = latest.summary

    authoritative_refutation = (
        latest.stance == EvidenceStance.REFUTES
        and latest.authority_score >= company_threshold
        and latest.source_type in {"公司", "官方"}
    )
    official_confirmation = (
        latest.stance == EvidenceStance.SUPPORTS
        and latest.authority_score >= official_threshold
        and latest.source_type == "官方"
    )

    if authoritative_refutation:
        return LifecycleStage.OFFICIAL_CLARIFIED, ClaimStatus.REFUTED, "权威来源发布澄清或否认，原始主张转为已证伪"
    if official_confirmation:
        return LifecycleStage.OFFICIAL_CONFIRMED, ClaimStatus.CONFIRMED, "官方证据确认事件，可信结论升级"
    if support_rows and refute_rows:
        return LifecycleStage.DISPUTED, ClaimStatus.DISPUTED, "支持与反驳证据并存，事件进入争议状态"
    if any(keyword in latest_text for keyword in lifecycle_cfg["resolution_keywords"]):
        return LifecycleStage.RESOLVED, ClaimStatus.RESOLVED, "出现处置完成证据，事件生命周期完结"
    if (
        any(keyword in latest_text for keyword in lifecycle_cfg["escalation_keywords"])
        and support_rows
    ):
        status = ClaimStatus.CONFIRMED if any(
            row.source_type == "官方" for row in support_rows
        ) else ClaimStatus.SUPPORTED
        return LifecycleStage.IMPACT_ESCALATED, status, "出现影响扩大信号，事件关注等级提升"
    if latest.source_type == "公司":
        status = ClaimStatus.SUPPORTED if latest.stance == EvidenceStance.SUPPORTS else ClaimStatus.UNVERIFIED
        return LifecycleStage.COMPANY_RESPONDED, status, "公司来源作出回应，补充事件证据"
    if len(independent_support) >= lifecycle_cfg["corroborated_source_count"]:
        return LifecycleStage.CORROBORATED, ClaimStatus.SUPPORTED, "多个独立来源提供一致支持证据"
    if evidence:
        return LifecycleStage.MEDIA_REPORTED, ClaimStatus.UNVERIFIED, "新增媒体或其他来源报道，结论仍待核实"
    return LifecycleStage.DISCOVERED, ClaimStatus.UNVERIFIED, "首次发现事件线索"


def _evolve_credibility(
    current: float,
    evidence: EvidenceRecord,
    is_new_source: bool,
    stage: LifecycleStage,
    lifecycle_cfg: dict,
) -> float:
    base_gain = lifecycle_cfg["base_evidence_gain"] * evidence.authority_score
    source_bonus = lifecycle_cfg["independent_source_bonus"] if is_new_source else 0.0

    if evidence.stance == EvidenceStance.SUPPORTS:
        score = current + base_gain + source_bonus
        if stage == LifecycleStage.OFFICIAL_CONFIRMED:
            score += lifecycle_cfg["official_confirmation_bonus"]
    elif evidence.stance == EvidenceStance.REFUTES:
        if stage == LifecycleStage.OFFICIAL_CLARIFIED:
            refutation_confidence = (
                evidence.authority_score * 0.7
                + lifecycle_cfg["clarification_confidence_bonus"]
            )
            score = max(current, refutation_confidence)
        else:
            score = current * lifecycle_cfg["dispute_confidence_factor"]
    else:
        score = current + base_gain * 0.35 + source_bonus * 0.35

    return round(min(1.0, max(0.0, score)), 4)
