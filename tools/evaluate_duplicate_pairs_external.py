from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventlens.config import load_settings
from eventlens.duplicate_pair_evaluation import (
    evaluate_external_duplicate_pairs,
    load_duplicate_pairs_jsonl,
    write_benchmark_report,
)
from eventlens.event_retrieval import OllamaEmbeddingClient


def main() -> None:
    scope = sys.argv[1] if len(sys.argv) > 1 else "company"
    if scope not in {"company", "industry"}:
        raise SystemExit("scope 必须是 company 或 industry")

    settings = load_settings(ROOT / "configs/app.yaml")
    train_report = json.loads(
        (ROOT / f"reports/duplicate_pair_benchmark_{scope}.json").read_text(
            encoding="utf-8"
        )
    )
    embedding = settings.event_retrieval.embedding
    evaluation = evaluate_external_duplicate_pairs(
        load_duplicate_pairs_jsonl(
            ROOT / f"artifacts/training_pairs/{scope}_pairs_test.jsonl"
        ),
        OllamaEmbeddingClient(
            base_url=embedding.base_url,
            model=embedding.model,
            timeout_seconds=embedding.timeout_seconds,
            batch_size=embedding.batch_size,
        ),
        title_threshold=train_report["title_similarity"]["calibration"]["threshold"],
        bge_threshold=train_report["bge_cosine"]["calibration"]["threshold"],
    )
    output = ROOT / f"reports/duplicate_pair_external_{scope}.json"
    write_benchmark_report(output, evaluation)
    print(evaluation.model_dump_json())


if __name__ == "__main__":
    main()
