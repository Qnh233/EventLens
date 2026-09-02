from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from eventlens.event_retrieval import EventSchemaIndex, RoutedArticleRecallResult
from eventlens.subject_routing import SubjectRouteResult


def predict_from_scores(
    classes: list[str] | np.ndarray,
    scores: np.ndarray,
    *,
    allowed_labels: Iterable[str] | None = None,
) -> str:
    """在允许标签集合内取最高分；集合为空时安全回退到全局最高分。"""

    labels = [str(label) for label in classes]
    row = np.asarray(scores, dtype=np.float64)
    if row.ndim != 1 or row.shape[0] != len(labels):
        raise ValueError("分类分数维度与 classes 不一致")

    allowed = set(allowed_labels or [])
    eligible = [index for index, label in enumerate(labels) if label in allowed]
    if not eligible:
        return labels[int(np.argmax(row))]
    winner = max(eligible, key=lambda index: float(row[index]))
    return labels[winner]


def subject_allowed_events(
    schema: EventSchemaIndex,
    route: SubjectRouteResult,
    *,
    scope: str,
    include_candidates_when_unresolved: bool,
) -> set[str]:
    by_code = schema.by_company_code if scope == "company" else schema.by_industry_code
    if route.accepted_subject_code:
        codes = [route.accepted_subject_code]
    elif include_candidates_when_unresolved:
        codes = [row.subject_code for row in route.candidates]
    else:
        codes = []
    return {
        definition.event_name
        for code in codes
        for definition in by_code.get(code, [])
    }


def recall_allowed_events(recall: RoutedArticleRecallResult, *, top_k: int) -> set[str]:
    return {row.event_name for row in recall.candidates[: max(1, top_k)]}


def constrain_predictions(
    classes: list[str] | np.ndarray,
    score_matrix: np.ndarray,
    routes: list[SubjectRouteResult],
    recalls: list[RoutedArticleRecallResult],
    schema: EventSchemaIndex,
    *,
    scope: str,
    policy: str,
    recall_top_k: int = 3,
) -> list[str]:
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != len(routes) or len(routes) != len(recalls):
        raise ValueError("约束分类输入数量不一致")

    output: list[str] = []
    for row_scores, route, recall in zip(scores, routes, recalls):
        global_prediction = predict_from_scores(classes, row_scores)
        if policy == "global":
            allowed: set[str] = set()
        elif policy == "hard_subject":
            allowed = subject_allowed_events(
                schema,
                route,
                scope=scope,
                include_candidates_when_unresolved=False,
            )
        elif policy == "subject_union":
            allowed = subject_allowed_events(
                schema,
                route,
                scope=scope,
                include_candidates_when_unresolved=True,
            )
        elif policy == "hard_subject_recall":
            subject_allowed = subject_allowed_events(
                schema,
                route,
                scope=scope,
                include_candidates_when_unresolved=False,
            )
            recall_allowed = recall_allowed_events(recall, top_k=recall_top_k)
            allowed = subject_allowed & recall_allowed if subject_allowed else set()
        elif policy in {"hard_subject_recall_fallback", "exact_subject_recall_fallback"}:
            if policy == "exact_subject_recall_fallback" and route.method != "exact_alias":
                allowed = set()
            else:
                subject_allowed = subject_allowed_events(
                    schema,
                    route,
                    scope=scope,
                    include_candidates_when_unresolved=False,
                )
                if not subject_allowed or global_prediction in subject_allowed:
                    allowed = set()
                else:
                    allowed = subject_allowed & recall_allowed_events(
                        recall, top_k=recall_top_k
                    )
        elif policy == "recall_topk":
            allowed = recall_allowed_events(recall, top_k=recall_top_k)
        else:
            raise ValueError(f"未知约束策略: {policy}")
        output.append(predict_from_scores(classes, row_scores, allowed_labels=allowed))
    return output
