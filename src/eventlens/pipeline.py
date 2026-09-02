from __future__ import annotations

from pathlib import Path
from typing import Any

from eventlens.baseline import TrainedBaseline, predict_articles
from eventlens.cluster import cluster_events
from eventlens.config import load_settings
from eventlens.credibility import build_alerts
from eventlens.evidence_control import (
    ClaimEvidenceBinding,
    EvidenceGateDecision,
    evaluate_evidence_controls,
)
from eventlens.event_retrieval import OllamaEmbeddingClient
from eventlens.io import write_jsonl
from eventlens.learning import LearningSignal, ObserverAgent, SkillRegistry, SkillRuntime
from eventlens.lifecycle import EventLifecycle, EventLifecycleLedger, build_event_lifecycles
from eventlens.schema import (
    AlertOutput,
    ArticleRecord,
    ClusterDecision,
    EventCluster,
    EventPrediction,
)
from eventlens.semantic_similarity import BgeSemanticPairScorer, SemanticPairScorer


def run_pipeline(
    articles: list[ArticleRecord],
    model: TrainedBaseline | None = None,
    model_config: dict | None = None,
    cluster_config: dict | None = None,
    credibility_config: dict | None = None,
    lifecycle_config: dict | None = None,
    learning_config: dict | None = None,
    skill_registry_path: str | Path | None = None,
    semantic_scorer: SemanticPairScorer | None = None,
) -> dict[str, list[Any]]:
    settings = load_settings()
    m_cfg = model_config or settings.model.model_dump()
    c_cfg = cluster_config or settings.cluster.model_dump()
    cr_cfg = credibility_config or settings.credibility.model_dump()
    lc_cfg = lifecycle_config or settings.lifecycle.model_dump()
    learn_cfg = learning_config or settings.learning.model_dump()

    predictions = predict_articles(articles, model=model, config=m_cfg)
    registry = SkillRegistry(skill_registry_path or settings.paths.skill_registry)
    predictions = SkillRuntime(registry.active_skills()).apply(articles, predictions)
    semantic_cfg = c_cfg.get("semantic", {"enabled": False})
    if semantic_cfg.get("enabled", False) and semantic_scorer is None:
        embedding = semantic_cfg["embedding"]
        semantic_scorer = BgeSemanticPairScorer(
            OllamaEmbeddingClient(
                base_url=embedding["base_url"],
                model=embedding["model"],
                timeout_seconds=embedding["timeout_seconds"],
                batch_size=embedding["batch_size"],
                num_gpu=embedding.get("num_gpu"),
            ),
            model=embedding["model"],
            cache_path=semantic_cfg.get("cache_path"),
        )
    cluster_decisions: list[ClusterDecision] = []
    clusters = cluster_events(
        articles,
        predictions,
        config=c_cfg,
        semantic_scorer=semantic_scorer,
        decision_sink=cluster_decisions,
    )
    alerts = build_alerts(articles, clusters, cfg=cr_cfg)
    lifecycles = build_event_lifecycles(
        articles,
        predictions,
        clusters,
        lifecycle_config=lc_cfg,
        credibility_config=cr_cfg,
    )
    alerts, claim_bindings, evidence_gates = evaluate_evidence_controls(
        lifecycles,
        alerts,
        settings.evidence_control.model_dump(),
    )
    observer = ObserverAgent(
        low_confidence_threshold=learn_cfg["low_confidence_threshold"],
        high_severity_threshold=learn_cfg["high_severity_threshold"],
    )
    learning_signals = observer.observe(lifecycles, alerts)
    return {
        "predictions": predictions,
        "clusters": clusters,
        "cluster_decisions": cluster_decisions,
        "alerts": alerts,
        "lifecycles": lifecycles,
        "claim_bindings": claim_bindings,
        "evidence_gates": evidence_gates,
        "learning_signals": learning_signals,
    }


def write_pipeline_outputs(
    output_dir: str | Path,
    result: dict[str, list[Any]],
    lifecycle_ledger_path: str | Path | None = None,
) -> None:
    out = Path(output_dir)
    write_jsonl(out / "article_event.jsonl", _typed(result["predictions"], EventPrediction))
    write_jsonl(out / "event_cluster.jsonl", _typed(result["clusters"], EventCluster))
    write_jsonl(
        out / "cluster_decision.jsonl",
        _typed(result.get("cluster_decisions", []), ClusterDecision),
    )
    write_jsonl(out / "alert_output.jsonl", _typed(result["alerts"], AlertOutput))
    write_jsonl(
        out / "claim_evidence.jsonl",
        _typed(result.get("claim_bindings", []), ClaimEvidenceBinding),
    )
    write_jsonl(
        out / "evidence_gate.jsonl",
        _typed(result.get("evidence_gates", []), EvidenceGateDecision),
    )
    lifecycles = _typed(result.get("lifecycles", []), EventLifecycle)
    signals = _typed(result.get("learning_signals", []), LearningSignal)
    if lifecycle_ledger_path:
        ledger = EventLifecycleLedger(lifecycle_ledger_path)
        previous = ledger.latest()
        lifecycles = [
            lifecycle.model_copy(
                update={
                    "version": previous[lifecycle.event_cluster_id].version + 1,
                    "created_at": previous[lifecycle.event_cluster_id].created_at,
                }
            )
            if lifecycle.event_cluster_id in previous
            else lifecycle
            for lifecycle in lifecycles
        ]
    write_jsonl(out / "event_lifecycle.jsonl", lifecycles)
    write_jsonl(out / "learning_signals.jsonl", signals)
    if lifecycle_ledger_path:
        ledger.append(lifecycles)


def _typed(rows: list[Any], expected_type: type) -> list[Any]:
    return [row for row in rows if isinstance(row, expected_type)]

