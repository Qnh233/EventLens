from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventlens.cluster_benchmark import (
    build_cluster_benchmark_dataset,
    run_cluster_grid_benchmark,
)
from eventlens.config import load_settings
from eventlens.duplicate_pairs import build_duplicate_cluster_groups
from eventlens.event_retrieval import OllamaEmbeddingClient
from eventlens.io import read_competition_labeled_excel, write_json
from eventlens.semantic_similarity import BgeSemanticPairScorer


def main() -> None:
    scope = sys.argv[1] if len(sys.argv) > 1 else "company"
    if scope not in {"company", "industry"}:
        raise SystemExit("scope 必须是 company 或 industry")
    settings = load_settings(ROOT / "configs/app.yaml")
    data = read_competition_labeled_excel(ROOT / settings.paths.tagged_train)
    pair_config = settings.duplicate_pairs
    groups = build_duplicate_cluster_groups(
        data,
        scope=scope,
        max_articles=settings.cluster_benchmark.max_articles,
        subject_lead_chars=pair_config.subject_lead_chars,
        min_subject_alias_chars=pair_config.min_subject_alias_chars,
    )
    dataset = build_cluster_benchmark_dataset(groups, scope=scope)
    semantic = settings.cluster.semantic
    embedding = semantic.embedding
    scorer = BgeSemanticPairScorer(
        OllamaEmbeddingClient(
            base_url=embedding.base_url,
            model=embedding.model,
            timeout_seconds=embedding.timeout_seconds,
            batch_size=embedding.batch_size,
            num_gpu=embedding.num_gpu,
        ),
        model=embedding.model,
        cache_path=ROOT / semantic.cache_path,
    )
    benchmark = settings.cluster_benchmark
    report = run_cluster_grid_benchmark(
        dataset,
        cluster_config=settings.cluster.model_dump(),
        semantic_scorer=scorer,
        candidate_thresholds=benchmark.candidate_thresholds,
        semantic_thresholds=benchmark.semantic_thresholds,
        top_ks=benchmark.top_ks,
        minimum_b_cubed_f1_gain=benchmark.minimum_b_cubed_f1_gain,
        minimum_pairwise_recall_gain=benchmark.minimum_pairwise_recall_gain,
    )
    output = ROOT / f"reports/cluster_benchmark_{scope}.json"
    write_json(output, report.model_dump())
    print(report.model_dump_json())


if __name__ == "__main__":
    main()
