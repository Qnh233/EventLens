from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eventlens.lifecycle import ClaimStatus, EventLifecycle
from eventlens.schema import AlertOutput, ArticleRecord, EventPrediction


class SkillStatus(str, Enum):
    CANDIDATE = "candidate"
    EVALUATED = "evaluated"
    SHADOW = "shadow"
    ACTIVE = "active"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class FeedbackRecord(BaseModel):
    feedback_id: str
    kind: str
    article_id: str | None = None
    event_cluster_id: str | None = None
    expected_event_type: str | None = None
    observed_event_type: str | None = None
    note: str = ""
    created_at: datetime
    reviewer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LearningSignal(BaseModel):
    signal_id: str
    event_cluster_id: str
    reason: str
    priority: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    suggested_action: str
    created_at: datetime


class SkillDefinition(BaseModel):
    skill_id: str
    name: str
    version: str = "1.0.0"
    status: SkillStatus = SkillStatus.CANDIDATE
    target_event_type: str
    observed_event_type: str | None = None
    trigger_terms: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    source_feedback_ids: list[str] = Field(default_factory=list)
    provenance_hash: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime
    evaluated_at: datetime | None = None
    shadow_started_at: datetime | None = None
    approved_at: datetime | None = None
    approved_by: str | None = None
    rejection_reason: str | None = None
    rolled_back_at: datetime | None = None
    rolled_back_by: str | None = None
    rollback_reason: str | None = None


class FlywheelResult(BaseModel):
    candidates: list[SkillDefinition] = Field(default_factory=list)
    promoted_count: int = 0
    shadow_count: int = 0
    rejected_count: int = 0


class SkillRegistry:
    """Skill 版本注册表。每次状态变化追加一条记录，保留完整审计历史。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, skill: SkillDefinition) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = skill.model_dump(mode="json")
        payload["registry_recorded_at"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def history(self) -> list[SkillDefinition]:
        if not self.path.exists():
            return []
        rows: list[SkillDefinition] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                payload.pop("registry_recorded_at", None)
                rows.append(SkillDefinition.model_validate(payload))
        return rows

    def latest(self) -> dict[str, SkillDefinition]:
        rows: dict[str, SkillDefinition] = {}
        for skill in self.history():
            rows[skill.skill_id] = skill
        return rows

    def active_skills(self) -> list[SkillDefinition]:
        return sorted(
            [skill for skill in self.latest().values() if skill.status == SkillStatus.ACTIVE],
            key=lambda skill: skill.skill_id,
        )

    def rollback(
        self,
        skill_id: str,
        *,
        reason: str,
        rolled_back_by: str,
    ) -> SkillDefinition:
        skill = self.latest().get(skill_id)
        if skill is None:
            raise KeyError(f"未知 Skill: {skill_id}")
        if skill.status not in {SkillStatus.ACTIVE, SkillStatus.SHADOW}:
            raise ValueError("只有 ACTIVE/SHADOW Skill 可以回滚")
        now = datetime.now(timezone.utc)
        rolled_back = skill.model_copy(
            update={
                "status": SkillStatus.ROLLED_BACK,
                "rolled_back_at": now,
                "rolled_back_by": rolled_back_by,
                "rollback_reason": reason,
            }
        )
        self.append(rolled_back)
        return rolled_back

    def export_active_skills(self, output_dir: str | Path) -> list[Path]:
        root = Path(output_dir)
        exported: list[Path] = []
        for skill in self.active_skills():
            skill_dir = root / f"{skill.name}-{skill.version}"
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "skill.json").write_text(
                json.dumps(skill.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (skill_dir / "SKILL.md").write_text(_render_skill_markdown(skill), encoding="utf-8")
            exported.append(skill_dir)
        return exported


class ObserverAgent:
    def __init__(self, low_confidence_threshold: float, high_severity_threshold: float):
        self.low_confidence_threshold = low_confidence_threshold
        self.high_severity_threshold = high_severity_threshold

    def observe(
        self,
        lifecycles: list[EventLifecycle],
        alerts: list[AlertOutput],
    ) -> list[LearningSignal]:
        alert_map = {alert.event_cluster_id: alert for alert in alerts}
        signals: list[LearningSignal] = []
        for lifecycle in lifecycles:
            reasons: list[str] = []
            priority = 0.0
            if (
                lifecycle.credibility_score < self.low_confidence_threshold
                and len(lifecycle.evidence) >= 2
            ):
                reasons.append("低置信事件")
                priority += 0.35
            if lifecycle.claim_status == ClaimStatus.DISPUTED:
                reasons.append("证据冲突")
                priority += 0.45
            if lifecycle.claim_status == ClaimStatus.REFUTED:
                reasons.append("官方澄清或事实反转")
                priority += 0.50
            alert = alert_map.get(lifecycle.event_cluster_id)
            if alert and alert.severity_score >= self.high_severity_threshold:
                reasons.append("高影响事件")
                priority += 0.25
            if not reasons:
                continue
            signal_key = f"{lifecycle.event_cluster_id}|{'|'.join(reasons)}|{lifecycle.version}"
            signal_id = f"LS-{hashlib.sha256(signal_key.encode('utf-8')).hexdigest()[:16]}"
            signals.append(
                LearningSignal(
                    signal_id=signal_id,
                    event_cluster_id=lifecycle.event_cluster_id,
                    reason="、".join(reasons),
                    priority=min(1.0, round(priority, 4)),
                    evidence_ids=[row.evidence_id for row in lifecycle.evidence],
                    suggested_action="进入人工复核并沉淀结构化反馈",
                    created_at=datetime.now(timezone.utc),
                )
            )
        return signals


class DiagnosisAgent:
    def diagnose(self, feedback: FeedbackRecord) -> str:
        mapping = {
            "wrong_type": "EVENT_TYPE_ERROR",
            "wrong_polarity": "POLARITY_ERROR",
            "over_merge": "COREFERENCE_OVER_MERGE",
            "missed_merge": "COREFERENCE_MISSED_MERGE",
            "wrong_credibility": "CREDIBILITY_ERROR",
        }
        return mapping.get(feedback.kind, "DOMAIN_EDGE_CASE")


class SkillCuratorAgent:
    def __init__(self, min_feedback_count: int):
        self.min_feedback_count = min_feedback_count
        self.diagnosis_agent = DiagnosisAgent()

    def distill(self, feedback: list[FeedbackRecord]) -> list[SkillDefinition]:
        grouped: dict[tuple[str, str, str], list[FeedbackRecord]] = defaultdict(list)
        for row in feedback:
            diagnosis = self.diagnosis_agent.diagnose(row)
            key = (
                diagnosis,
                row.expected_event_type or "",
                row.observed_event_type or "",
            )
            grouped[key].append(row)

        candidates: list[SkillDefinition] = []
        for (diagnosis, expected, observed), rows in sorted(grouped.items()):
            if len(rows) < self.min_feedback_count or not expected:
                continue
            trigger_terms = _extract_trigger_terms(rows, expected, observed)
            if not trigger_terms:
                continue
            seed = f"{diagnosis}|{expected}|{observed}|{'|'.join(trigger_terms)}"
            digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            feedback_ids = sorted(row.feedback_id for row in rows)
            provenance_hash = hashlib.sha256(
                "|".join(feedback_ids).encode("utf-8")
            ).hexdigest()
            skill_id = f"SK-{digest[:16]}"
            skill_name = f"eventlens-event-boundary-{digest[:8]}"
            trigger_text = "、".join(trigger_terms)
            instruction = (
                f"当文本命中“{trigger_text}”且当前事件类型为“{observed or '其他'}”时，"
                f"将事件类型调整为“{expected}”，并记录 Skill 决策轨迹。"
            )
            candidates.append(
                SkillDefinition(
                    skill_id=skill_id,
                    name=skill_name,
                    target_event_type=expected,
                    observed_event_type=observed or None,
                    trigger_terms=trigger_terms,
                    instructions=[instruction],
                    negative_examples=[
                        "仅词面相似但不满足事件事实条件时不得强制改判。",
                        "多个已审批 Skill 冲突时保持原预测并转人工复核。",
                    ],
                    source_feedback_ids=feedback_ids,
                    provenance_hash=provenance_hash,
                    created_at=datetime.now(timezone.utc),
                )
            )
        return candidates


class EvaluationAgent:
    def __init__(self, min_macro_f1_gain: float, max_critical_error_regression: float):
        self.min_macro_f1_gain = min_macro_f1_gain
        self.max_critical_error_regression = max_critical_error_regression

    def evaluate(
        self,
        baseline_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
    ) -> tuple[bool, str, dict[str, float]]:
        baseline_f1 = float(baseline_metrics.get("macro_f1", 0.0))
        candidate_f1 = float(candidate_metrics.get("macro_f1", 0.0))
        baseline_critical = float(baseline_metrics.get("critical_error_rate", 0.0))
        candidate_critical = float(candidate_metrics.get("critical_error_rate", 0.0))
        f1_gain = candidate_f1 - baseline_f1
        critical_regression = candidate_critical - baseline_critical
        metrics = {
            "baseline_macro_f1": baseline_f1,
            "candidate_macro_f1": candidate_f1,
            "macro_f1_gain": f1_gain,
            "baseline_critical_error_rate": baseline_critical,
            "candidate_critical_error_rate": candidate_critical,
            "critical_error_regression": critical_regression,
        }
        if f1_gain < self.min_macro_f1_gain:
            return False, "Macro-F1 提升未达到发布门槛", metrics
        if critical_regression > self.max_critical_error_regression:
            return False, "关键错误率出现不可接受退化", metrics
        return True, "离线指标满足发布门槛", metrics


class ReleaseAgent:
    def __init__(self, require_human_approval: bool):
        self.require_human_approval = require_human_approval

    def release(
        self,
        skill: SkillDefinition,
        evaluation_passed: bool,
        evaluation_reason: str,
        metrics: dict[str, float],
        human_approved: bool,
        approved_by: str | None,
    ) -> SkillDefinition:
        now = datetime.now(timezone.utc)
        if not evaluation_passed:
            return skill.model_copy(
                update={
                    "status": SkillStatus.REJECTED,
                    "metrics": metrics,
                    "evaluated_at": now,
                    "rejection_reason": evaluation_reason,
                }
            )
        if self.require_human_approval and not human_approved:
            return skill.model_copy(
                update={
                    "status": SkillStatus.SHADOW,
                    "metrics": metrics,
                    "evaluated_at": now,
                    "shadow_started_at": now,
                    "rejection_reason": "影子评估通过，等待人工审批",
                }
            )
        return skill.model_copy(
            update={
                "status": SkillStatus.ACTIVE,
                "metrics": metrics,
                "evaluated_at": now,
                "approved_at": now,
                "approved_by": approved_by or "system-policy",
                "rejection_reason": None,
            }
        )


class FlywheelOrchestrator:
    def __init__(
        self,
        registry: SkillRegistry,
        min_feedback_count: int,
        min_macro_f1_gain: float,
        max_critical_error_regression: float,
        require_human_approval: bool,
    ):
        self.registry = registry
        self.curator = SkillCuratorAgent(min_feedback_count)
        self.evaluator = EvaluationAgent(min_macro_f1_gain, max_critical_error_regression)
        self.releaser = ReleaseAgent(require_human_approval)

    def run(
        self,
        feedback: list[FeedbackRecord],
        baseline_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        human_approved: bool,
        approved_by: str | None = None,
    ) -> FlywheelResult:
        output: list[SkillDefinition] = []
        promoted = 0
        shadow = 0
        rejected = 0
        for candidate in self.curator.distill(feedback):
            self.registry.append(candidate)
            passed, reason, metrics = self.evaluator.evaluate(
                baseline_metrics,
                candidate_metrics,
            )
            released = self.releaser.release(
                candidate,
                passed,
                reason,
                metrics,
                human_approved,
                approved_by,
            )
            self.registry.append(released)
            output.append(released)
            promoted += int(released.status == SkillStatus.ACTIVE)
            shadow += int(released.status == SkillStatus.SHADOW)
            rejected += int(released.status == SkillStatus.REJECTED)
        return FlywheelResult(
            candidates=output,
            promoted_count=promoted,
            shadow_count=shadow,
            rejected_count=rejected,
        )


class SkillRuntime:
    """只应用 ACTIVE Skill；所有修正写入预测的审计轨迹。"""

    def __init__(self, skills: list[SkillDefinition]):
        self.skills = sorted(
            [skill for skill in skills if skill.status == SkillStatus.ACTIVE],
            key=lambda skill: skill.skill_id,
        )

    def apply(
        self,
        articles: list[ArticleRecord],
        predictions: list[EventPrediction],
    ) -> list[EventPrediction]:
        article_map = {article.article_id: article for article in articles}
        corrected: list[EventPrediction] = []
        for prediction in predictions:
            article = article_map.get(prediction.article_id)
            if article is None:
                corrected.append(prediction)
                continue
            text = f"{article.title} {article.content} {article.impact_analysis or ''}"
            current = prediction
            matched: list[SkillDefinition] = []
            for skill in self.skills:
                observed_matches = (
                    not skill.observed_event_type
                    or current.event_type == skill.observed_event_type
                )
                trigger_matches = any(term in text for term in skill.trigger_terms)
                if observed_matches and trigger_matches:
                    matched.append(skill)
            if len({skill.target_event_type for skill in matched}) > 1:
                trace = list(current.decision_trace)
                trace.append("多个已审批 Skill 给出冲突结论，保持原预测并建议人工复核")
                corrected.append(current.model_copy(update={"decision_trace": trace}))
                continue
            for skill in matched[:1]:
                skill_ids = list(current.applied_skill_ids)
                trace = list(current.decision_trace)
                skill_ids.append(skill.skill_id)
                trace.append(
                    f"命中已审批 Skill {skill.skill_id}，事件类型由“{current.event_type}”调整为“{skill.target_event_type}”"
                )
                current = current.model_copy(
                    update={
                        "has_event": True,
                        "event_type": skill.target_event_type,
                        "applied_skill_ids": skill_ids,
                        "decision_trace": trace,
                    }
                )
            corrected.append(current)
        return corrected


def load_feedback_jsonl(path: str | Path) -> list[FeedbackRecord]:
    rows: list[FeedbackRecord] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(FeedbackRecord.model_validate(json.loads(line)))
    return rows


def load_metrics_json(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in payload.items()}


def _extract_trigger_terms(
    rows: list[FeedbackRecord],
    expected: str,
    observed: str,
) -> list[str]:
    explicit_terms: list[str] = []
    for row in rows:
        explicit_terms.extend(str(term) for term in row.metadata.get("trigger_terms", []))
    if explicit_terms:
        return _deduplicate_terms(explicit_terms)[:8]

    marker_pattern = re.compile(r"(?:被误判|误判为|识别成|应识别为|应为|不是)")
    candidates: list[str] = []
    for row in rows:
        note = re.sub(r"\s+", "", row.note)
        prefix = marker_pattern.split(note, maxsplit=1)[0]
        prefix = prefix.replace(expected, "").replace(observed, "")
        prefix = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", prefix)
        if 2 <= len(prefix) <= 20:
            candidates.append(prefix)
    if candidates:
        counts: dict[str, int] = defaultdict(int)
        for candidate in candidates:
            counts[candidate] += 1
        ranked = sorted(counts, key=lambda term: (-counts[term], -len(term), term))
        return ranked[:8]

    words: list[str] = []
    for row in rows:
        words.extend(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,12}", row.note))
    stop_terms = {expected, observed, "事件", "错误", "误判", "识别", "调整"}
    return [term for term in _deduplicate_terms(words) if term not in stop_terms][:8]


def _deduplicate_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _render_skill_markdown(skill: SkillDefinition) -> str:
    trigger_text = "、".join(skill.trigger_terms)
    feedback_text = "、".join(skill.source_feedback_ids)
    instructions = "\n".join(f"{index}. {text}" for index, text in enumerate(skill.instructions, 1))
    negatives = "\n".join(f"- {text}" for text in skill.negative_examples)
    return (
        "---\n"
        f"name: {skill.name}\n"
        f"description: 修正 EventLens 中“{skill.target_event_type}”的稳定边界。"
        f"当文本包含{trigger_text}，或分析事件分类边界、复盘历史误判时使用。\n"
        "---\n\n"
        f"# {skill.target_event_type} 边界经验\n\n"
        "## 使用范围\n\n"
        f"- 目标事件类型：{skill.target_event_type}\n"
        f"- 原错误类型：{skill.observed_event_type or '未限定'}\n"
        f"- 触发词：{trigger_text}\n\n"
        "## 决策步骤\n\n"
        f"{instructions}\n\n"
        "## 安全边界\n\n"
        f"{negatives}\n\n"
        "## 审计来源\n\n"
        f"- Skill ID：{skill.skill_id}\n"
        f"- 版本：{skill.version}\n"
        f"- 人工审批人：{skill.approved_by or '未审批'}\n"
        f"- 来源反馈：{feedback_text}\n"
        f"- Provenance Hash：{skill.provenance_hash or '未生成'}\n"
    )
