from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventlens.config import load_settings
from eventlens.duplicate_pair_evaluation import load_duplicate_pairs_jsonl
from eventlens.duplicate_pair_reranker import evaluate_reranker_stability
from eventlens.event_retrieval import OllamaEmbeddingClient


def main() -> None:
    settings = load_settings(ROOT / "configs/app.yaml")
    embedding = settings.event_retrieval.embedding
    benchmark = json.loads(
        (ROOT / "reports/duplicate_pair_benchmark_company.json").read_text(
            encoding="utf-8"
        )
    )
    report = evaluate_reranker_stability(
        load_duplicate_pairs_jsonl(
            ROOT / "artifacts/training_pairs/company_pairs.jsonl"
        ),
        load_duplicate_pairs_jsonl(
            ROOT / "artifacts/training_pairs/company_pairs_test.jsonl"
        ),
        OllamaEmbeddingClient(
            base_url=embedding.base_url,
            model=embedding.model,
            timeout_seconds=embedding.timeout_seconds,
            batch_size=embedding.batch_size,
        ),
        bge_threshold=benchmark["bge_cosine"]["calibration"]["threshold"],
        seeds=[7, 19, 42, 73, 101],
        calibration_ratio=settings.duplicate_pair_evaluation.calibration_ratio,
    )
    output = ROOT / "reports/duplicate_pair_reranker_stability_company.json"
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json())


if __name__ == "__main__":
    main()
