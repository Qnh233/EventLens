from __future__ import annotations

from pathlib import Path
from typing import Any

from eventlens.baseline import TrainedBaseline, predict_articles
from eventlens.cluster import cluster_events
from eventlens.config import load_settings
from eventlens.credibility import build_alerts
from eventlens.io import write_jsonl
from eventlens.schema import AlertOutput, ArticleRecord, EventCluster, EventPrediction


def run_pipeline(
    articles: list[ArticleRecord],
    model: TrainedBaseline | None = None,
    model_config: dict | None = None,
    cluster_config: dict | None = None,
    credibility_config: dict | None = None,
) -> dict[str, list[Any]]:
    settings = load_settings()
    m_cfg = model_config or settings.model.model_dump()
    c_cfg = cluster_config or settings.cluster.model_dump()
    cr_cfg = credibility_config or settings.credibility.model_dump()

    predictions = predict_articles(articles, model=model, config=m_cfg)
    clusters = cluster_events(articles, predictions, config=c_cfg)
    alerts = build_alerts(articles, clusters, cfg=cr_cfg)
    return {"predictions": predictions, "clusters": clusters, "alerts": alerts}


def write_pipeline_outputs(output_dir: str | Path, result: dict[str, list[Any]]) -> None:
    out = Path(output_dir)
    write_jsonl(out / "article_event.jsonl", _typed(result["predictions"], EventPrediction))
    write_jsonl(out / "event_cluster.jsonl", _typed(result["clusters"], EventCluster))
    write_jsonl(out / "alert_output.jsonl", _typed(result["alerts"], AlertOutput))


def _typed(rows: list[Any], expected_type: type) -> list[Any]:
    return [row for row in rows if isinstance(row, expected_type)]

