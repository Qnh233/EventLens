from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel


class RunValidationReport(BaseModel):
    passed: bool
    prediction_count: int
    event_count: int
    subject_nonempty_count: int
    subject_nonempty_rate: float
    subject_methods: dict[str, int]
    cluster_count: int
    clustered_article_count: int
    multi_article_cluster_count: int
    max_cluster_size: int
    alert_count: int
    lifecycle_count: int
    learning_signal_count: int
    evidence_gate_count: int = 0
    evidence_gate_pass_rate: float = 0.0
    blocked_alert_count: int = 0
    claim_count: int = 0
    claim_evidence_coverage: float = 0.0
    unsupported_high_risk_claim_count: int = 0
    alert_levels: dict[str, int]
    learning_reasons: dict[str, int]
    errors: list[str]


def validate_run_output_dir(path: str | Path) -> RunValidationReport:
    root = Path(path)
    predictions = _load_jsonl(root / "article_event.jsonl")
    clusters = _load_jsonl(root / "event_cluster.jsonl")
    alerts = _load_jsonl(root / "alert_output.jsonl")
    lifecycles = _load_jsonl(root / "event_lifecycle.jsonl")
    signals = _load_jsonl(root / "learning_signals.jsonl")
    gates = _load_jsonl_optional(root / "evidence_gate.jsonl")
    claims = _load_jsonl_optional(root / "claim_evidence.jsonl")

    errors: list[str] = []
    prediction_ids = [str(row.get("article_id", "")) for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        errors.append("article_event 存在重复 article_id")
    event_ids = {
        str(row.get("article_id", ""))
        for row in predictions
        if bool(row.get("has_event"))
    }
    cluster_ids = [str(row.get("event_cluster_id", "")) for row in clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        errors.append("event_cluster 存在重复 event_cluster_id")
    cluster_id_set = set(cluster_ids)
    clustered_article_ids = {
        str(article_id)
        for row in clusters
        for article_id in row.get("article_ids", [])
    }
    if clustered_article_ids != event_ids:
        errors.append("事件文章与事件簇 article_id 覆盖不一致")
    if any(str(row.get("event_cluster_id", "")) not in cluster_id_set for row in alerts):
        errors.append("alert_output 存在无对应事件簇的引用")
    if any(
        str(row.get("event_cluster_id", "")) not in cluster_id_set for row in lifecycles
    ):
        errors.append("event_lifecycle 存在无对应事件簇的引用")
    if len(alerts) != len(clusters):
        errors.append("每个事件簇应对应一条 alert_output")
    if len(lifecycles) != len(clusters):
        errors.append("每个事件簇应对应一条 event_lifecycle")
    blocked_alerts = [row for row in alerts if row.get("delivery_allowed") is False]
    if gates:
        alert_gate_clusters = {
            str(row.get("event_cluster_id", ""))
            for row in gates
            if row.get("gate_type") == "alert_delivery"
        }
        if alert_gate_clusters != cluster_id_set:
            errors.append("alert delivery gate 与事件簇覆盖不一致")
    if claims:
        claim_clusters = {str(row.get("event_cluster_id", "")) for row in claims}
        if not claim_clusters.issubset(cluster_id_set):
            errors.append("claim_evidence 存在无对应事件簇的引用")

    subject_nonempty = sum(bool(row.get("event_subject")) for row in predictions)
    cluster_sizes = [len(row.get("article_ids", [])) for row in clusters]
    return RunValidationReport(
        passed=not errors,
        prediction_count=len(predictions),
        event_count=len(event_ids),
        subject_nonempty_count=subject_nonempty,
        subject_nonempty_rate=round(subject_nonempty / max(1, len(predictions)), 6),
        subject_methods=dict(Counter(str(row.get("event_subject_method", "")) for row in predictions)),
        cluster_count=len(clusters),
        clustered_article_count=len(clustered_article_ids),
        multi_article_cluster_count=sum(size > 1 for size in cluster_sizes),
        max_cluster_size=max(cluster_sizes, default=0),
        alert_count=len(alerts),
        lifecycle_count=len(lifecycles),
        learning_signal_count=len(signals),
        evidence_gate_count=len(gates),
        evidence_gate_pass_rate=round(
            sum(bool(row.get("passed")) for row in gates) / max(1, len(gates)), 6
        ),
        blocked_alert_count=len(blocked_alerts),
        claim_count=len(claims),
        claim_evidence_coverage=round(
            sum(bool(row.get("supported")) for row in claims) / max(1, len(claims)), 6
        ),
        unsupported_high_risk_claim_count=sum(
            1
            for row in claims
            if not bool(row.get("supported")) and row.get("kind") == "inference"
        ),
        alert_levels=dict(Counter(str(row.get("risk_level", "")) for row in alerts)),
        learning_reasons=dict(Counter(str(row.get("reason", "")) for row in signals)),
        errors=errors,
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_jsonl_optional(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _load_jsonl(path)
