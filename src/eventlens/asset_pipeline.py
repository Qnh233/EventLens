from __future__ import annotations

from pathlib import Path
from typing import Any

from eventlens.baseline import TrainedBaseline, predict_articles
from eventlens.candidate_clustering import cluster_candidate_events
from eventlens.candidate_edges import SubjectTimeCandidateEdge
from eventlens.config import load_settings
from eventlens.credibility import build_alerts
from eventlens.evidence_control import evaluate_evidence_controls
from eventlens.event_retrieval import RoutedArticleRecallResult
from eventlens.learning import ObserverAgent, SkillRegistry, SkillRuntime
from eventlens.lifecycle import build_event_lifecycles
from eventlens.schema import ArticleRecord, EventPrediction
from eventlens.subject_routing import SubjectRouteResult


def run_asset_pipeline(
    articles: list[ArticleRecord],
    vectors,
    routes: list[SubjectRouteResult],
    recalls: list[RoutedArticleRecallResult],
    edges: list[SubjectTimeCandidateEdge],
    *,
    scope: str,
    model: TrainedBaseline,
    skill_registry_path: str | Path | None = None,
) -> dict[str, list[Any]]:
    """复用离线向量/主体/事件候选资产，执行比赛主链路，不重复 BGE 编码。"""

    if scope not in {"company", "industry"}:
        raise ValueError("scope 必须是 company 或 industry")
    article_ids = [row.article_id for row in articles]
    if not (
        len(articles) == len(vectors) == len(routes) == len(recalls)
        and article_ids == [row.article_id for row in routes]
        and article_ids == [row.article_id for row in recalls]
    ):
        raise ValueError("文章、向量、主体路由和事件召回必须严格同序")

    settings = load_settings()
    predictions = predict_articles(articles, model=model, config=settings.model.model_dump())
    registry = SkillRegistry(skill_registry_path or settings.paths.skill_registry)
    predictions = SkillRuntime(registry.active_skills()).apply(articles, predictions)
    enriched_articles, enriched_predictions = _enrich_subjects(
        articles,
        predictions,
        routes,
        scope=scope,
    )
    clusters, cluster_decisions = cluster_candidate_events(
        enriched_articles,
        enriched_predictions,
        article_ids,
        vectors,
        edges,
        recalls,
        similarity_threshold=settings.cluster.semantic.similarity_threshold,
    )
    alerts = build_alerts(
        enriched_articles,
        clusters,
        cfg=settings.credibility.model_dump(),
    )
    lifecycles = build_event_lifecycles(
        enriched_articles,
        enriched_predictions,
        clusters,
        lifecycle_config=settings.lifecycle.model_dump(),
        credibility_config=settings.credibility.model_dump(),
    )
    alerts, claim_bindings, evidence_gates = evaluate_evidence_controls(
        lifecycles,
        alerts,
        settings.evidence_control.model_dump(),
    )
    observer = ObserverAgent(
        low_confidence_threshold=settings.learning.low_confidence_threshold,
        high_severity_threshold=settings.learning.high_severity_threshold,
    )
    learning_signals = observer.observe(lifecycles, alerts)
    return {
        "articles": enriched_articles,
        "predictions": enriched_predictions,
        "clusters": clusters,
        "cluster_decisions": cluster_decisions,
        "alerts": alerts,
        "lifecycles": lifecycles,
        "claim_bindings": claim_bindings,
        "evidence_gates": evidence_gates,
        "learning_signals": learning_signals,
    }


def _enrich_subjects(
    articles: list[ArticleRecord],
    predictions: list[EventPrediction],
    routes: list[SubjectRouteResult],
    *,
    scope: str,
) -> tuple[list[ArticleRecord], list[EventPrediction]]:
    enriched_articles: list[ArticleRecord] = []
    enriched_predictions: list[EventPrediction] = []
    for article, prediction, route in zip(articles, predictions, routes, strict=True):
        candidate = route.candidates[0] if route.candidates else None
        subject_name = route.accepted_subject_name or (candidate.subject_name if candidate else "")
        subject_code = route.accepted_subject_code or (candidate.subject_code if candidate else "")
        subject_confidence = 1.0 if route.method == "exact_alias" else route.top1_score
        trace = [
            *prediction.decision_trace,
            f"subject_route:{route.method}",
            f"subject_confidence:{subject_confidence:.6f}",
        ]
        enriched_predictions.append(
            prediction.model_copy(
                update={
                    "event_subject": subject_name or None,
                    "event_subject_confidence": round(subject_confidence, 6),
                    "event_subject_method": route.method,
                    "impact_target": subject_name or prediction.impact_target,
                    "decision_trace": trace,
                }
            )
        )
        article_update: dict[str, object] = {
            "polarity_label": prediction.event_polarity,
        }
        if scope == "company":
            article_update.update({"entity": subject_name, "trading_code": subject_code})
        else:
            article_update.update({"industry": subject_name, "industry_code": subject_code})
        enriched_articles.append(article.model_copy(update=article_update))
    return enriched_articles, enriched_predictions
