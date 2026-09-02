from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from eventlens.evidence_control import EvidenceGateDecision


class RuntimeState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STOPPED = "stopped"


class DependencyHealth(BaseModel):
    name: str
    healthy: bool
    required: bool = True
    fallback_available: bool = False


class RuntimeSnapshot(BaseModel):
    queue_depth: int = Field(ge=0)
    active_workers: int = Field(ge=0)
    consecutive_failures: int = Field(default=0, ge=0)
    process_alive: bool = True
    run_valid: bool = True
    dependencies: list[DependencyHealth] = Field(default_factory=list)


class RuntimeAction(BaseModel):
    action: str
    priority: int = Field(ge=0, le=100)
    reason: str
    target_workers: int | None = None
    retry_after_seconds: int | None = None


class CollectionRequest(BaseModel):
    event_cluster_id: str
    reason: str
    preferred_source_types: list[str] = Field(
        default_factory=lambda: ["官方", "公司", "主流财经媒体"]
    )
    priority: int = Field(default=80, ge=0, le=100)


class RuntimePlan(BaseModel):
    state: RuntimeState
    actions: list[RuntimeAction]
    collection_requests: list[CollectionRequest]
    unsafe_continue: bool = False


class RuntimeController:
    """轻量运行控制面：只做扩缩、恢复、降级、停止和补采决策。"""

    def __init__(self, config: dict):
        self.config = config

    def plan(
        self,
        snapshot: RuntimeSnapshot,
        evidence_gates: list[EvidenceGateDecision] | None = None,
    ) -> RuntimePlan:
        actions: list[RuntimeAction] = []
        requests = _collection_requests(evidence_gates or [])
        stop_after = int(self.config["stop_after_failures"])
        degraded_after = int(self.config["degraded_after_failures"])

        broken_required = [
            row
            for row in snapshot.dependencies
            if row.required and not row.healthy and not row.fallback_available
        ]
        fallback_dependencies = [
            row
            for row in snapshot.dependencies
            if not row.healthy and row.fallback_available
        ]

        if broken_required or snapshot.consecutive_failures >= stop_after or not snapshot.run_valid:
            reason = (
                "关键依赖不可用且无 fallback"
                if broken_required
                else "连续失败达到停止门槛"
                if snapshot.consecutive_failures >= stop_after
                else "输出完整性门禁失败"
            )
            actions.append(RuntimeAction(action="stop", priority=100, reason=reason))
            return RuntimePlan(
                state=RuntimeState.STOPPED,
                actions=actions,
                collection_requests=requests,
            )

        if not snapshot.process_alive:
            actions.append(
                RuntimeAction(
                    action="restart",
                    priority=95,
                    reason="检测到进程退出，从最近检查点恢复",
                    retry_after_seconds=self._backoff(snapshot.consecutive_failures),
                )
            )

        if fallback_dependencies:
            actions.append(
                RuntimeAction(
                    action="fallback",
                    priority=90,
                    reason="依赖异常，切换到已验证的缓存/规则保守路径："
                    + ",".join(row.name for row in fallback_dependencies),
                )
            )

        if snapshot.consecutive_failures >= degraded_after:
            actions.append(
                RuntimeAction(
                    action="degrade",
                    priority=85,
                    reason="连续失败达到 degraded 门槛，暂停非关键增强能力",
                )
            )
        elif snapshot.consecutive_failures > 0 and snapshot.process_alive:
            actions.append(
                RuntimeAction(
                    action="retry",
                    priority=75,
                    reason="任务失败但未达到熔断门槛",
                    retry_after_seconds=self._backoff(snapshot.consecutive_failures),
                )
            )

        self._plan_scaling(snapshot, actions)
        if requests:
            actions.append(
                RuntimeAction(
                    action="collect_evidence",
                    priority=80,
                    reason=f"{len(requests)} 个事件未通过高风险证据门禁，生成定向补采请求",
                )
            )

        if not actions:
            actions.append(RuntimeAction(action="continue", priority=10, reason="系统健康且无调度动作"))

        state = (
            RuntimeState.DEGRADED
            if any(row.action in {"degrade", "fallback", "restart"} for row in actions)
            else RuntimeState.HEALTHY
        )
        unsafe_continue = any(row.action == "continue" for row in actions) and (
            not snapshot.process_alive or bool(broken_required) or not snapshot.run_valid
        )
        return RuntimePlan(
            state=state,
            actions=sorted(actions, key=lambda row: -row.priority),
            collection_requests=requests,
            unsafe_continue=unsafe_continue,
        )

    def _plan_scaling(self, snapshot: RuntimeSnapshot, actions: list[RuntimeAction]) -> None:
        min_workers = int(self.config["min_workers"])
        max_workers = int(self.config["max_workers"])
        if (
            snapshot.queue_depth >= int(self.config["scale_up_queue_depth"])
            and snapshot.active_workers < max_workers
            and snapshot.consecutive_failures == 0
        ):
            actions.append(
                RuntimeAction(
                    action="scale_up",
                    priority=60,
                    reason="队列积压超过扩容阈值",
                    target_workers=min(max_workers, max(min_workers, snapshot.active_workers + 1)),
                )
            )
        elif (
            snapshot.queue_depth <= int(self.config["scale_down_queue_depth"])
            and snapshot.active_workers > min_workers
        ):
            actions.append(
                RuntimeAction(
                    action="scale_down",
                    priority=40,
                    reason="队列低水位，回收空闲 worker",
                    target_workers=max(min_workers, snapshot.active_workers - 1),
                )
            )

    def _backoff(self, failures: int) -> int:
        backoff = [int(row) for row in self.config["retry_backoff_seconds"]]
        return backoff[min(max(failures - 1, 0), len(backoff) - 1)]


def _collection_requests(gates: list[EvidenceGateDecision]) -> list[CollectionRequest]:
    requests: list[CollectionRequest] = []
    seen: set[str] = set()
    for gate in gates:
        if gate.gate_type != "alert_delivery" or gate.passed or gate.event_cluster_id in seen:
            continue
        seen.add(gate.event_cluster_id)
        requests.append(
            CollectionRequest(
                event_cluster_id=gate.event_cluster_id,
                reason=gate.reason,
            )
        )
    return requests
