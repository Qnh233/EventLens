from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eventlens.config import load_settings
from eventlens.duplicate_pair_evaluation import load_duplicate_pairs_jsonl
from eventlens.duplicate_pair_reranker import (
    evaluate_lightweight_reranker,
    write_reranker_report,
)
from eventlens.event_retrieval import OllamaEmbeddingClient


def main() -> None:
    scope = sys.argv[1] if len(sys.argv) > 1 else "company"
    if scope != "company":
        raise SystemExit("当前仅允许 company；行业同主体覆盖不足，暂不训练")

    settings = load_settings(ROOT / "configs/app.yaml")
    embedding = settings.event_retrieval.embedding
    benchmark = json.loads(
        (ROOT / "reports/duplicate_pair_benchmark_company.json").read_text(
            encoding="utf-8"
        )
    )
    report = evaluate_lightweight_reranker(
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
        calibration_ratio=settings.duplicate_pair_evaluation.calibration_ratio,
        seed=settings.duplicate_pair_evaluation.seed,
    )
    write_reranker_report(
        ROOT / "reports/duplicate_pair_reranker_company.json", report
    )
    print(report.model_dump_json())


if __name__ == "__main__":
    main()
