from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventlens.cluster import cluster_events
from eventlens.config import load_settings
from eventlens.event_retrieval import OllamaEmbeddingClient
from eventlens.schema import ArticleRecord, EventPrediction
from eventlens.semantic_similarity import BgeSemanticPairScorer


class FailEmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("cache miss")


def main() -> None:
    settings = load_settings(ROOT / "configs/app.yaml")
    config = settings.cluster.model_dump()
    config["threshold"] = 0.99
    config["semantic"]["enabled"] = True
    config["semantic"]["candidate_threshold"] = 0.4
    cache_path = ROOT / "artifacts/cache/semantic_cluster_smoke.sqlite3"
    cache_path.unlink(missing_ok=True)

    articles = [
        ArticleRecord(
            article_id="SMOKE-1",
            title="示例公司新能源电池工厂正式开工",
            content="公司宣布新能源电池工厂项目进入建设阶段。",
            publish_time=datetime(2026, 8, 1),
            trading_code="000001",
            entity="示例公司",
        ),
        ArticleRecord(
            article_id="SMOKE-2",
            title="新能源电池项目启动建设",
            content="示例公司的电池工厂已经正式开工建设。",
            publish_time=datetime(2026, 8, 2),
            trading_code="000001",
            entity="示例公司",
        ),
    ]
    predictions = [
        EventPrediction(
            article_id="SMOKE-1",
            has_event=True,
            event_type="重大项目投资建设",
            classifier_confidence=0.8,
            evidence_sentence="新能源电池工厂正式开工",
        ),
        EventPrediction(
            article_id="SMOKE-2",
            has_event=True,
            event_type="重大项目投资建设",
            classifier_confidence=0.8,
            evidence_sentence="电池工厂项目启动建设",
        ),
    ]
    embedding = config["semantic"]["embedding"]
    first_scorer = BgeSemanticPairScorer(
        OllamaEmbeddingClient(
            base_url=embedding["base_url"],
            model=embedding["model"],
            timeout_seconds=embedding["timeout_seconds"],
            batch_size=embedding["batch_size"],
        ),
        model=embedding["model"],
        cache_path=cache_path,
    )
    first_decisions = []
    first_clusters = cluster_events(
        articles,
        predictions,
        config=config,
        semantic_scorer=first_scorer,
        decision_sink=first_decisions,
    )

    second_scorer = BgeSemanticPairScorer(
        FailEmbeddingClient(),
        model=embedding["model"],
        cache_path=cache_path,
    )
    second_decisions = []
    second_clusters = cluster_events(
        articles,
        predictions,
        config=config,
        semantic_scorer=second_scorer,
        decision_sink=second_decisions,
    )
    print(
        json.dumps(
            {
                "first_cluster_count": len(first_clusters),
                "second_cluster_count": len(second_clusters),
                "semantic_score": first_decisions[0].semantic_score,
                "cache_reused_without_embedding_call": True,
                "cache_path": str(cache_path.relative_to(ROOT)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
