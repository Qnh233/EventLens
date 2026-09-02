from __future__ import annotations

from collections import Counter

from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score

from eventlens.event_retrieval import RoutedArticleRecallResult
from eventlens.llm_agent import AgentDecision, EventChangeVerifier, EventExpertAgent
from eventlens.schema import ArticleRecord, EventPrediction
from eventlens.subject_routing import SubjectRouteResult


class AgentEvaluationReport(BaseModel):
    sample_count: int
    baseline_accuracy: float
    baseline_macro_f1: float
    agent_accuracy: float
    agent_macro_f1: float
    answered_count: int
    answered_accuracy: float
    valid_rate: float
    changed_count: int
    corrected_count: int
    harmed_count: int
    abstain_count: int
    collection_request_count: int
    average_steps: float
    average_latency_ms: float
    tool_usage: dict[str, int]


class AgentShadowSummary(BaseModel):
    sample_count: int
    valid_rate: float
    changed_suggestion_count: int
    abstain_count: int
    collection_request_count: int
    verifier_accepted_count: int
    verifier_rejected_count: int
    average_steps: float
    average_latency_ms: float
    p95_latency_ms: int
    tool_usage: dict[str, int]


def select_agent_case_indices(
    predictions: list[EventPrediction],
    routes: list[SubjectRouteResult],
    recalls: list[RoutedArticleRecallResult],
    *,
    max_samples: int,
    confidence_max: float,
    subject_margin_max: float,
) -> list[int]:
    if not (len(predictions) == len(routes) == len(recalls)):
        raise ValueError("Agent case selector 输入数量不一致")
    scored: list[tuple[float, float, int]] = []
    for index, (prediction, route, recall) in enumerate(zip(predictions, routes, recalls)):
        # Event Expert 只在主体已通过 hard-route 后介入；主体未决属于另一类任务，
        # 否则会在错误主体的事件候选集合里做“精确的错误判断”。
        if route.accepted_subject_code is None:
            continue
        candidate_names = [row.event_name for row in recall.candidates]
        outside_candidates = bool(candidate_names) and prediction.event_type not in candidate_names
        low_confidence = prediction.classifier_confidence <= confidence_max
        low_subject_margin = route.top1_margin <= subject_margin_max
        if not (outside_candidates or low_confidence or low_subject_margin):
            continue
        score = (
            2.0 * int(outside_candidates)
            + 1.0 * int(low_confidence)
            + 1.0 * int(low_subject_margin)
        )
        scored.append((score, -prediction.classifier_confidence, index))
    scored.sort(reverse=True)
    return [index for _, _, index in scored[:max_samples]]


def run_agent_cases(
    agent: EventExpertAgent,
    articles: list[ArticleRecord],
    predictions: list[EventPrediction],
    routes: list[SubjectRouteResult],
    recalls: list[RoutedArticleRecallResult],
    indices: list[int],
) -> list[AgentDecision]:
    return [
        agent.run(articles[index], predictions[index], routes[index], recalls[index])
        for index in indices
    ]


def verify_agent_decisions(
    verifier: EventChangeVerifier,
    articles: list[ArticleRecord],
    predictions: list[EventPrediction],
    recalls: list[RoutedArticleRecallResult],
    indices: list[int],
    decisions: list[AgentDecision],
) -> list[AgentDecision]:
    if len(indices) != len(decisions):
        raise ValueError("Verifier 输入数量不一致")
    return [
        verifier.verify(
            articles[index],
            predictions[index],
            recalls[index],
            decision,
        )
        for index, decision in zip(indices, decisions)
    ]


def summarize_agent_shadow(decisions: list[AgentDecision]) -> AgentShadowSummary:
    count = len(decisions)
    tool_usage: Counter[str] = Counter()
    latencies: list[int] = []
    for row in decisions:
        for call in row.tool_calls:
            tool_usage[call.action] += 1
        latencies.append(row.latency_ms + row.verifier_latency_ms)
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
    p95 = ordered[p95_index] if ordered else 0
    return AgentShadowSummary(
        sample_count=count,
        valid_rate=round(sum(row.valid for row in decisions) / max(1, count), 6),
        changed_suggestion_count=sum(
            row.final_event is not None and row.final_event != row.baseline_event
            for row in decisions
        ),
        abstain_count=sum(row.final_event is None for row in decisions),
        collection_request_count=sum(row.requested_collection for row in decisions),
        verifier_accepted_count=sum(row.verifier_accepted is True for row in decisions),
        verifier_rejected_count=sum(row.verifier_accepted is False for row in decisions),
        average_steps=round(sum(row.steps for row in decisions) / max(1, count), 3),
        average_latency_ms=round(sum(latencies) / max(1, count), 3),
        p95_latency_ms=p95,
        tool_usage=dict(tool_usage),
    )


def evaluate_agent_decisions(
    articles: list[ArticleRecord],
    predictions: list[EventPrediction],
    indices: list[int],
    decisions: list[AgentDecision],
) -> AgentEvaluationReport:
    if len(indices) != len(decisions):
        raise ValueError("Agent decision 数量与选择样本不一致")
    truth = [str(articles[index].event_label or "") for index in indices]
    if any(not label for label in truth):
        raise ValueError("Agent 外部评测要求 event_label")
    baseline = [predictions[index].event_type for index in indices]
    agent_pred = [decision.final_event or "__ABSTAIN__" for decision in decisions]
    answered_indices = [i for i, decision in enumerate(decisions) if decision.final_event is not None]
    changed = 0
    corrected = 0
    harmed = 0
    tool_usage: Counter[str] = Counter()
    for local_index, decision in enumerate(decisions):
        for call in decision.tool_calls:
            tool_usage[call.action] += 1
        if decision.final_event is not None and decision.final_event != baseline[local_index]:
            changed += 1
            baseline_ok = baseline[local_index] == truth[local_index]
            agent_ok = decision.final_event == truth[local_index]
            corrected += int(not baseline_ok and agent_ok)
            harmed += int(baseline_ok and not agent_ok)
    answered_truth = [truth[i] for i in answered_indices]
    answered_pred = [agent_pred[i] for i in answered_indices]
    count = len(indices)
    return AgentEvaluationReport(
        sample_count=count,
        baseline_accuracy=round(accuracy_score(truth, baseline), 6) if count else 0.0,
        baseline_macro_f1=round(
            f1_score(truth, baseline, average="macro", zero_division=0), 6
        ) if count else 0.0,
        agent_accuracy=round(accuracy_score(truth, agent_pred), 6) if count else 0.0,
        agent_macro_f1=round(
            f1_score(truth, agent_pred, average="macro", zero_division=0), 6
        ) if count else 0.0,
        answered_count=len(answered_indices),
        answered_accuracy=round(
            accuracy_score(answered_truth, answered_pred), 6
        ) if answered_indices else 0.0,
        valid_rate=round(sum(row.valid for row in decisions) / max(1, count), 6),
        changed_count=changed,
        corrected_count=corrected,
        harmed_count=harmed,
        abstain_count=sum(row.final_event is None for row in decisions),
        collection_request_count=sum(row.requested_collection for row in decisions),
        average_steps=round(sum(row.steps for row in decisions) / max(1, count), 3),
        average_latency_ms=round(
            sum(row.latency_ms for row in decisions) / max(1, count), 3
        ),
        tool_usage=dict(tool_usage),
    )
