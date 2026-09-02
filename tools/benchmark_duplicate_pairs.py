from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventlens.config import load_settings
from eventlens.duplicate_pair_evaluation import (
    benchmark_duplicate_pairs,
    load_duplicate_pairs_jsonl,
    write_benchmark_report,
)
from eventlens.event_retrieval import OllamaEmbeddingClient


def main() -> None:
    scope = sys.argv[1] if len(sys.argv) > 1 else "company"
    if scope not in {"company", "industry"}:
        raise SystemExit("scope 必须是 company 或 industry")

    settings = load_settings(ROOT / "configs/app.yaml")
    embedding = settings.event_retrieval.embedding
    input_path = ROOT / f"artifacts/training_pairs/{scope}_pairs.jsonl"
    output_path = ROOT / f"reports/duplicate_pair_benchmark_{scope}.json"
    benchmark = benchmark_duplicate_pairs(
        load_duplicate_pairs_jsonl(input_path),
        OllamaEmbeddingClient(
            base_url=embedding.base_url,
            model=embedding.model,
            timeout_seconds=embedding.timeout_seconds,
            batch_size=embedding.batch_size,
        ),
        calibration_ratio=settings.duplicate_pair_evaluation.calibration_ratio,
        seed=settings.duplicate_pair_evaluation.seed,
    )
    write_benchmark_report(output_path, benchmark)
    print(benchmark.model_dump_json())


if __name__ == "__main__":
    main()
