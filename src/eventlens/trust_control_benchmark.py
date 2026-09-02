from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from eventlens.evidence_control import evaluate_evidence_controls
from eventlens.learning import (
    ReleaseAgent,
    SkillDefinition,
    SkillRegistry,
    SkillRuntime,
    SkillStatus,
)
from eventlens.lifecycle import (
    ClaimStatus,
    EventLifecycle,
    EvidenceRecord,
    EvidenceStance,
    LifecycleStage,
)
from eventlens.schema import (
    AlertOutput,
    ArticleRecord,
    CredibilityBreakdown,
    EventPrediction,
)


class TrustScenarioResult(BaseModel):
    name: str
    expected: str
    actual: str
    passed: bool


class TrustControlBenchmarkReport(BaseModel):
    evidence_scenario_count: int
    evidence_gate_match_rate: float
    skill_scenario_count: int
    skill_governance_match_rate: float
    scenarios: list[TrustScenarioResult]


def benchmark_trust_controls(evidence_config: dict) -> TrustControlBenchmarkReport:
    scenarios: list[TrustScenarioResult] = []
    alert = _high_risk_alert()
    evidence_cases = [
        ("high_risk_single_source", [_evidence("E1", "media-1")], "blocked"),
        (
            "high_risk_two_sources",
            [_evidence("E1", "media-1"), _evidence("E2", "media-2")],
            "allowed",
        ),
        (
            "high_risk_official_source",
            [_evidence("E1", "official", source_type="官方", authority=1.0)],
            "allowed",
        ),
    ]
    for name, evidence, expected in evidence_cases:
        guarded, _, _ = evaluate_evidence_controls(
            [_lifecycle(evidence)], [alert], evidence_config
        )
        actual = "allowed" if guarded[0].delivery_allowed else "blocked"
        scenarios.append(
            TrustScenarioResult(
                name=name,
                expected=expected,
                actual=actual,
                passed=actual == expected,
            )
        )
    guarded, _, _ = evaluate_evidence_controls([], [alert], evidence_config)
    actual = "allowed" if guarded[0].delivery_allowed else "blocked"
    scenarios.append(
        TrustScenarioResult(
            name="missing_lifecycle",
            expected="blocked",
            actual=actual,
            passed=actual == "blocked",
        )
    )
    evidence_count = len(scenarios)

    releaser = ReleaseAgent(require_human_approval=True)
    candidate = _skill()
    shadow = releaser.release(
        candidate,
        True,
        "pass",
        {"macro_f1_gain": 0.02},
        False,
        None,
    )
    scenarios.append(_status_scenario("skill_shadow_before_approval", shadow, SkillStatus.SHADOW))
    active = releaser.release(
        candidate,
        True,
        "pass",
        {"macro_f1_gain": 0.02},
        True,
        "analyst",
    )
    scenarios.append(_status_scenario("skill_active_after_approval", active, SkillStatus.ACTIVE))
    rejected = releaser.release(
        candidate,
        False,
        "metric regression",
        {"macro_f1_gain": -0.01},
        True,
        "analyst",
    )
    scenarios.append(_status_scenario("skill_rejected_on_metric_regression", rejected, SkillStatus.REJECTED))

    article = ArticleRecord(article_id="A1", title="监管问询函", content="测试")
    prediction = EventPrediction(article_id="A1", has_event=True, event_type="业绩预告")
    shadow_prediction = SkillRuntime([shadow]).apply([article], [prediction])[0]
    scenarios.append(
        TrustScenarioResult(
            name="shadow_skill_does_not_modify_prediction",
            expected="业绩预告",
            actual=shadow_prediction.event_type,
            passed=shadow_prediction.event_type == "业绩预告",
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        registry = SkillRegistry(Path(tmp) / "registry.jsonl")
        registry.append(active)
        registry.rollback(active.skill_id, reason="regression", rolled_back_by="analyst")
        rollback_safe = not registry.active_skills()
    scenarios.append(
        TrustScenarioResult(
            name="rollback_removes_active_skill",
            expected="inactive",
            actual="inactive" if rollback_safe else "active",
            passed=rollback_safe,
        )
    )

    evidence_results = scenarios[:evidence_count]
    skill_results = scenarios[evidence_count:]
    return TrustControlBenchmarkReport(
        evidence_scenario_count=len(evidence_results),
        evidence_gate_match_rate=round(
            sum(row.passed for row in evidence_results) / len(evidence_results), 6
        ),
        skill_scenario_count=len(skill_results),
        skill_governance_match_rate=round(
            sum(row.passed for row in skill_results) / len(skill_results), 6
        ),
        scenarios=scenarios,
    )


def _high_risk_alert() -> AlertOutput:
    return AlertOutput(
        alert_id="ALT-1",
        risk_level="高风险",
        event_cluster_id="EC-1",
        company="测试公司",
        event_type="监管处罚",
        event_summary="测试公司发生监管处罚",
        credibility_score=0.8,
        severity_score=0.8,
        credibility_breakdown=CredibilityBreakdown(
            source_authority=0.8,
            multi_source_consistency=0.5,
            official_endorsement=0.0,
            content_completeness=1.0,
            recency_score=1.0,
        ),
        evidence_sources=[],
        push_reason="测试",
        related_article_count=1,
    )


def _evidence(
    evidence_id: str,
    source_key: str,
    *,
    source_type: str = "主流财经媒体",
    authority: float = 0.75,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        article_id=evidence_id,
        source=source_key,
        source_type=source_type,
        independent_source_key=source_key,
        stance=EvidenceStance.SUPPORTS,
        authority_score=authority,
        summary="确认监管处罚",
        content_hash=evidence_id,
    )


def _lifecycle(evidence: list[EvidenceRecord]) -> EventLifecycle:
    now = datetime.now(timezone.utc)
    return EventLifecycle(
        event_cluster_id="EC-1",
        event_type="监管处罚",
        stage=LifecycleStage.MEDIA_REPORTED,
        claim_status=ClaimStatus.SUPPORTED,
        credibility_score=0.8,
        evidence=evidence,
        created_at=now,
        updated_at=now,
    )


def _skill() -> SkillDefinition:
    return SkillDefinition(
        skill_id="SK-1",
        name="test-skill",
        target_event_type="监管处罚",
        observed_event_type="业绩预告",
        trigger_terms=["问询函"],
        source_feedback_ids=["FB-1", "FB-2", "FB-3"],
        provenance_hash="test-provenance",
        created_at=datetime.now(timezone.utc),
    )


def _status_scenario(
    name: str, skill: SkillDefinition, expected: SkillStatus
) -> TrustScenarioResult:
    return TrustScenarioResult(
        name=name,
        expected=expected.value,
        actual=skill.status.value,
        passed=skill.status == expected,
    )
