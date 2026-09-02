from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from eventlens.event_retrieval import EventSchemaIndex, RoutedArticleRecallResult
from eventlens.subject_routing import SubjectRouteResult


@dataclass(frozen=True)
class CandidateFeatureRow:
    event_name: str
    features: dict[str, float | str]


def build_candidate_feature_rows(
    classes: list[str] | np.ndarray,
    svc_scores: np.ndarray,
    recall: RoutedArticleRecallResult,
    route: SubjectRouteResult,
    schema: EventSchemaIndex,
    *,
    scope: str,
    svc_top_k: int = 5,
    bge_top_k: int = 5,
    label_counts: dict[str, int] | None = None,
) -> list[CandidateFeatureRow]:
    """合并 SVC/BGE 候选，并只提取生产时可获得的排序特征。"""

    labels = [str(value) for value in classes]
    scores = np.asarray(svc_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.shape[0] != len(labels):
        raise ValueError("SVC 分数维度与 classes 不一致")
    if scope not in {"company", "industry"}:
        raise ValueError("scope 必须是 company 或 industry")

    order = np.argsort(-scores)
    rank_by_name = {labels[index]: rank for rank, index in enumerate(order, start=1)}
    score_by_name = {label: float(scores[index]) for index, label in enumerate(labels)}
    top_score = float(scores[int(order[0])]) if len(order) else 0.0

    bge_rows = recall.candidates[: max(0, bge_top_k)]
    bge_score_by_name: dict[str, float] = {}
    bge_rank_by_name: dict[str, int] = {}
    for rank, candidate in enumerate(bge_rows, start=1):
        name = str(candidate.event_name)
        bge_score_by_name[name] = max(
            float(candidate.score), bge_score_by_name.get(name, -1.0)
        )
        bge_rank_by_name.setdefault(name, rank)
    bge_top_score = max(bge_score_by_name.values(), default=0.0)

    names: list[str] = []
    seen: set[str] = set()
    for index in order[: max(1, svc_top_k)]:
        name = labels[int(index)]
        if name not in seen:
            names.append(name)
            seen.add(name)
    for candidate in bge_rows:
        name = str(candidate.event_name)
        if name not in seen:
            names.append(name)
            seen.add(name)

    by_code = schema.by_company_code if scope == "company" else schema.by_industry_code
    accepted_events = _events_for_codes(
        by_code, [route.accepted_subject_code] if route.accepted_subject_code else []
    )
    route_codes = [candidate.subject_code for candidate in route.candidates]
    if route.accepted_subject_code and route.accepted_subject_code not in route_codes:
        route_codes.insert(0, route.accepted_subject_code)
    route_events = _events_for_codes(by_code, [code for code in route_codes if code])

    counts = label_counts or {}
    fallback_svc = min(score_by_name.values(), default=0.0) - 1.0
    output: list[CandidateFeatureRow] = []
    for name in names:
        svc_score = score_by_name.get(name, fallback_svc)
        svc_rank = rank_by_name.get(name)
        bge_score = bge_score_by_name.get(name, -1.0)
        bge_rank = bge_rank_by_name.get(name)
        in_svc = svc_rank is not None and svc_rank <= max(1, svc_top_k)
        in_bge = bge_rank is not None
        output.append(
            CandidateFeatureRow(
                event_name=name,
                features={
                    "event_name": name,
                    "route_method": route.method,
                    "svc_score": svc_score,
                    "svc_gap_top1": svc_score - top_score,
                    "svc_rank_inv": 1.0 / svc_rank if svc_rank else 0.0,
                    "svc_top1": float(svc_rank == 1),
                    "svc_in_topk": float(in_svc),
                    "bge_score": bge_score,
                    "bge_gap_top1": bge_score - bge_top_score if in_bge else -1.0,
                    "bge_rank_inv": 1.0 / bge_rank if bge_rank else 0.0,
                    "bge_top1": float(bge_rank == 1),
                    "bge_present": float(in_bge),
                    "in_both": float(in_svc and in_bge),
                    "schema_accepted": float(name in accepted_events),
                    "schema_any_route": float(name in route_events),
                    "route_top1_score": float(route.top1_score),
                    "route_top1_margin": float(route.top1_margin),
                    "train_count_log1p": math.log1p(max(0, int(counts.get(name, 0)))),
                },
            )
        )
    return output


def choose_best_candidate(names: list[str], probabilities: list[float]) -> str:
    if not names or len(names) != len(probabilities):
        raise ValueError("候选名称与概率必须非空且等长")
    return names[max(range(len(names)), key=lambda index: probabilities[index])]


def _events_for_codes(by_code: dict, codes: list[str]) -> set[str]:
    return {
        definition.event_name
        for code in codes
        for definition in by_code.get(code, [])
    }
