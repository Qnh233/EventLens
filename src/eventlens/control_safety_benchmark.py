from __future__ import annotations

from pydantic import BaseModel

from eventlens.evidence_control import EvidenceGateDecision
from eventlens.runtime_control import (
    DependencyHealth,
    RuntimeController,
    RuntimeSnapshot,
)


class ControlScenarioResult(BaseModel):
    name: str
    expected_action: str
    actual_actions: list[str]
    passed: bool
    unsafe_continue: bool


class ControlSafetyBenchmarkReport(BaseModel):
    scenario_count: int
    passed_count: int
    action_match_rate: float
    unsafe_continue_rate: float
    collection_trigger_rate: float
    scenarios: list[ControlScenarioResult]


def benchmark_control_safety(config: dict) -> ControlSafetyBenchmarkReport:
    controller = RuntimeController(config)
    scenarios = [
        ("healthy", RuntimeSnapshot(queue_depth=100, active_workers=1), [], "continue"),
        ("queue_scale_up", RuntimeSnapshot(queue_depth=800, active_workers=1), [], "scale_up"),
        ("queue_scale_down", RuntimeSnapshot(queue_depth=10, active_workers=3), [], "scale_down"),
        ("single_failure", RuntimeSnapshot(queue_depth=100, active_workers=1, consecutive_failures=1), [], "retry"),
        ("process_crash", RuntimeSnapshot(queue_depth=100, active_workers=1, process_alive=False, consecutive_failures=1), [], "restart"),
        ("degraded", RuntimeSnapshot(queue_depth=100, active_workers=1, consecutive_failures=3), [], "degrade"),
        ("circuit_break", RuntimeSnapshot(queue_depth=100, active_workers=1, consecutive_failures=5), [], "stop"),
        (
            "dependency_fallback",
            RuntimeSnapshot(
                queue_depth=100,
                active_workers=1,
                dependencies=[DependencyHealth(name="bge", healthy=False, fallback_available=True)],
            ),
            [],
            "fallback",
        ),
        (
            "dependency_hard_stop",
            RuntimeSnapshot(
                queue_depth=100,
                active_workers=1,
                dependencies=[DependencyHealth(name="model", healthy=False, fallback_available=False)],
            ),
            [],
            "stop",
        ),
        ("invalid_output", RuntimeSnapshot(queue_depth=0, active_workers=1, run_valid=False), [], "stop"),
        (
            "evidence_gap",
            RuntimeSnapshot(queue_depth=100, active_workers=1),
            [
                EvidenceGateDecision(
                    gate_id="G1",
                    event_cluster_id="EC-1",
                    gate_type="alert_delivery",
                    target="高风险",
                    passed=False,
                    reason="高风险证据不足",
                )
            ],
            "collect_evidence",
        ),
    ]
    results: list[ControlScenarioResult] = []
    collection_hits = 0
    for name, snapshot, gates, expected in scenarios:
        plan = controller.plan(snapshot, gates)
        actions = [row.action for row in plan.actions]
        passed = expected in actions
        collection_hits += int(bool(plan.collection_requests))
        results.append(
            ControlScenarioResult(
                name=name,
                expected_action=expected,
                actual_actions=actions,
                passed=passed,
                unsafe_continue=plan.unsafe_continue,
            )
        )
    return ControlSafetyBenchmarkReport(
        scenario_count=len(results),
        passed_count=sum(row.passed for row in results),
        action_match_rate=round(sum(row.passed for row in results) / len(results), 6),
        unsafe_continue_rate=round(
            sum(row.unsafe_continue for row in results) / len(results), 6
        ),
        collection_trigger_rate=round(collection_hits / len(results), 6),
        scenarios=results,
    )
