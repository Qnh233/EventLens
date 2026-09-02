from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from eventlens.event_retrieval import RoutedArticleRecallResult
from eventlens.subject_routing import SubjectRouteResult


class HardExample(BaseModel):
    article_id: str
    scope: str
    reasons: list[str]
    priority: float = Field(ge=0.0)
    accepted_subject_code: str | None = None
    subject_top1_score: float
    subject_top1_margin: float
    event_top1_score: float | None = None
    event_top1_margin: float | None = None


def build_hard_examples(
    routes: list[SubjectRouteResult],
    recalls: list[RoutedArticleRecallResult],
    *,
    subject_margin_threshold: float,
    event_margin_threshold: float,
    include_subject_rejection_signal: bool = True,
    max_examples: int = 5000,
) -> list[HardExample]:
    """从拒识、主体近边界和事件近边界样本构造可审计难例池。"""

    if len(routes) != len(recalls):
        raise ValueError("主体路由数量与事件召回数量不一致")
    if max_examples < 1:
        raise ValueError("max_examples 必须大于 0")

    output: list[HardExample] = []
    for route, recall in zip(routes, recalls):
        if route.article_id != recall.article_id or route.scope != recall.scope:
            raise ValueError("主体路由与事件召回顺序不一致")

        reasons: list[str] = []
        priority = 0.0
        if include_subject_rejection_signal and route.accepted_subject_code is None:
            reasons.append("subject_rejected")
            priority += 2.0
        if route.top1_margin < subject_margin_threshold:
            reasons.append("subject_low_margin")
            priority += subject_margin_threshold - route.top1_margin

        event_top1_score: float | None = None
        event_top1_margin: float | None = None
        if recall.candidates:
            event_top1_score = recall.candidates[0].score
            second_score = recall.candidates[1].score if len(recall.candidates) > 1 else -1.0
            event_top1_margin = event_top1_score - second_score
            if len(recall.candidates) > 1 and event_top1_margin < event_margin_threshold:
                reasons.append("event_low_margin")
                priority += event_margin_threshold - event_top1_margin
        else:
            reasons.append("event_no_candidate")
            priority += 2.0

        if reasons:
            output.append(
                HardExample(
                    article_id=route.article_id,
                    scope=route.scope,
                    reasons=reasons,
                    priority=round(priority, 6),
                    accepted_subject_code=route.accepted_subject_code,
                    subject_top1_score=route.top1_score,
                    subject_top1_margin=route.top1_margin,
                    event_top1_score=event_top1_score,
                    event_top1_margin=(
                        round(event_top1_margin, 6)
                        if event_top1_margin is not None
                        else None
                    ),
                )
            )

    output.sort(key=lambda row: (-row.priority, row.article_id))
    return output[:max_examples]


def load_routed_recalls_jsonl(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[RoutedArticleRecallResult]:
    rows: list[RoutedArticleRecallResult] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(RoutedArticleRecallResult.model_validate(json.loads(line)))
                if limit is not None and len(rows) >= limit:
                    break
    return rows
