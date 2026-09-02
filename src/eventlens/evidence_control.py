from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from eventlens.lifecycle import (
    ClaimStatus,
    EventLifecycle,
    EvidenceStance,
    LifecycleStage,
)
from eventlens.schema import AlertOutput


class ClaimKind(str, Enum):
    FACT = "fact"
    INFERENCE = "inference"
    UNVERIFIED = "unverified"


class ClaimEvidenceBinding(BaseModel):
    claim_id: str
    event_cluster_id: str
    claim_text: str
    kind: ClaimKind
    evidence_ids: list[str] = Field(default_factory=list)
    supported: bool
    reason: str


class EvidenceGateDecision(BaseModel):
    gate_id: str
    event_cluster_id: str
    gate_type: str
    target: str
    passed: bool
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str


def evaluate_evidence_controls(
    lifecycles: list[EventLifecycle],
    alerts: list[AlertOutput],
    config: dict,
) -> tuple[list[AlertOutput], list[ClaimEvidenceBinding], list[EvidenceGateDecision]]:
    lifecycle_map = {row.event_cluster_id: row for row in lifecycles}
    bindings: list[ClaimEvidenceBinding] = []
    gates: list[EvidenceGateDecision] = []
    guarded_alerts: list[AlertOutput] = []

    for alert in alerts:
        lifecycle = lifecycle_map.get(alert.event_cluster_id)
        if lifecycle is None:
            gate = EvidenceGateDecision(
                gate_id=f"GATE-{alert.event_cluster_id}-alert",
                event_cluster_id=alert.event_cluster_id,
                gate_type="alert_delivery",
                target=alert.risk_level,
                passed=False,
                reason="缺少生命周期记录，阻断预警发布",
            )
            gates.append(gate)
            guarded_alerts.append(
                alert.model_copy(
                    update={
                        "delivery_allowed": False,
                        "delivery_gate_reason": gate.reason,
                        "claim_evidence_coverage": 0.0,
                    }
                )
            )
            continue

        lifecycle_gates = evaluate_lifecycle_gates(lifecycle, config)
        gates.extend(lifecycle_gates)
        alert_bindings = bind_alert_claims(alert, lifecycle)
        bindings.extend(alert_bindings)
        alert_gate = evaluate_alert_delivery_gate(alert, lifecycle, config)
        gates.append(alert_gate)
        coverage = sum(row.supported for row in alert_bindings) / max(1, len(alert_bindings))
        guarded_alerts.append(
            alert.model_copy(
                update={
                    "delivery_allowed": alert_gate.passed,
                    "delivery_gate_reason": alert_gate.reason,
                    "claim_evidence_coverage": round(coverage, 6),
                }
            )
        )
    return guarded_alerts, bindings, gates


def bind_alert_claims(
    alert: AlertOutput,
    lifecycle: EventLifecycle,
) -> list[ClaimEvidenceBinding]:
    evidence_ids = [row.evidence_id for row in lifecycle.evidence]
    has_evidence = bool(evidence_ids)
    high_confidence = lifecycle.claim_status in {
        ClaimStatus.SUPPORTED,
        ClaimStatus.CONFIRMED,
        ClaimStatus.RESOLVED,
    }
    return [
        ClaimEvidenceBinding(
            claim_id=f"CLM-{alert.event_cluster_id}-event",
            event_cluster_id=alert.event_cluster_id,
            claim_text=f"{alert.company or '相关主体'}发生{alert.event_type}事件",
            kind=ClaimKind.FACT if high_confidence else ClaimKind.UNVERIFIED,
            evidence_ids=evidence_ids,
            supported=has_evidence,
            reason="事件主张绑定生命周期证据" if has_evidence else "无可追溯证据",
        ),
        ClaimEvidenceBinding(
            claim_id=f"CLM-{alert.event_cluster_id}-credibility",
            event_cluster_id=alert.event_cluster_id,
            claim_text=f"事件可信度评分为{alert.credibility_score:.4f}",
            kind=ClaimKind.INFERENCE,
            evidence_ids=evidence_ids,
            supported=has_evidence,
            reason="评分由已绑定证据的来源与一致性规则计算" if has_evidence else "评分缺少输入证据",
        ),
        ClaimEvidenceBinding(
            claim_id=f"CLM-{alert.event_cluster_id}-action",
            event_cluster_id=alert.event_cluster_id,
            claim_text=f"建议处置等级为{alert.risk_level}",
            kind=ClaimKind.INFERENCE,
            evidence_ids=evidence_ids,
            supported=has_evidence,
            reason="处置等级由可信度×严重性矩阵计算" if has_evidence else "处置建议缺少证据基础",
        ),
    ]


def evaluate_lifecycle_gates(
    lifecycle: EventLifecycle,
    config: dict,
) -> list[EvidenceGateDecision]:
    gates: list[EvidenceGateDecision] = []
    evidence_by_id = {row.evidence_id: row for row in lifecycle.evidence}
    for snapshot in lifecycle.snapshots:
        evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in snapshot.evidence_ids
            if evidence_id in evidence_by_id
        ]
        passed, reason = _stage_gate(snapshot.stage, evidence, config)
        gates.append(
            EvidenceGateDecision(
                gate_id=f"GATE-{lifecycle.event_cluster_id}-life-{snapshot.sequence}",
                event_cluster_id=lifecycle.event_cluster_id,
                gate_type="lifecycle_transition",
                target=snapshot.stage.value,
                passed=passed,
                evidence_ids=[row.evidence_id for row in evidence],
                reason=reason,
            )
        )
    return gates


def evaluate_alert_delivery_gate(
    alert: AlertOutput,
    lifecycle: EventLifecycle,
    config: dict,
) -> EvidenceGateDecision:
    evidence = lifecycle.evidence
    evidence_ids = [row.evidence_id for row in evidence]
    independent_sources = {row.independent_source_key for row in evidence}
    official = any(
        row.source_type == "官方"
        and row.authority_score >= float(config["official_bypass_authority"])
        for row in evidence
    )
    high_risk = alert.risk_level in set(config["high_risk_levels"])
    if not evidence:
        passed = False
        reason = "无证据，Proof-or-Stop 阻断预警"
    elif high_risk and not (
        official
        or len(independent_sources)
        >= int(config["minimum_high_risk_independent_sources"])
    ):
        passed = False
        reason = "高风险预警缺少官方证据或足够独立信源，转自主补采/人工核验"
    else:
        passed = True
        reason = "证据门禁通过"
    return EvidenceGateDecision(
        gate_id=f"GATE-{alert.event_cluster_id}-alert",
        event_cluster_id=alert.event_cluster_id,
        gate_type="alert_delivery",
        target=alert.risk_level,
        passed=passed,
        evidence_ids=evidence_ids,
        reason=reason,
    )


def _stage_gate(stage: LifecycleStage, evidence: list, config: dict) -> tuple[bool, str]:
    if not evidence:
        return False, "状态跃迁无证据"
    if stage == LifecycleStage.CORROBORATED:
        sources = {row.independent_source_key for row in evidence if row.stance == EvidenceStance.SUPPORTS}
        ok = len(sources) >= int(config.get("minimum_high_risk_independent_sources", 2))
        return ok, "多源佐证满足独立信源门槛" if ok else "多源佐证缺少足够独立支持信源"
    if stage == LifecycleStage.OFFICIAL_CONFIRMED:
        ok = any(row.source_type == "官方" and row.stance == EvidenceStance.SUPPORTS for row in evidence)
        return ok, "存在官方支持证据" if ok else "官方确认状态缺少官方支持证据"
    if stage == LifecycleStage.DISPUTED:
        stances = {row.stance for row in evidence}
        ok = EvidenceStance.SUPPORTS in stances and EvidenceStance.REFUTES in stances
        return ok, "支持与反驳证据并存" if ok else "争议状态缺少双向证据"
    if stage == LifecycleStage.OFFICIAL_CLARIFIED:
        ok = any(
            row.stance == EvidenceStance.REFUTES
            and row.source_type in {"官方", "公司"}
            for row in evidence
        )
        return ok, "存在权威澄清/反驳证据" if ok else "澄清状态缺少权威反驳证据"
    return True, "状态至少绑定一条证据"
