from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_llm_teacher_gold import build_client, candidate_descriptions
from benchmark_retrieval_augmented_llm_teacher import (
    _group_ids,
    _normalize_rows,
    _retrieved_exemplars,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.env import load_env_file
from eventlens.event_retrieval import EventSchemaIndex
from eventlens.io import read_competition_labeled_excel
from eventlens.llm_teacher import CandidateChangeVerifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--teacher-details", required=True)
    parser.add_argument("--min-teacher-confidence", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exemplar-content-chars", type=int, default=500)
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", required=True)
    args = parser.parse_args()

    load_env_file()
    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in train]
    manifest, vectors = load_exported_vectors(args.train_embeddings_dir)
    if manifest.article_count != len(train):
        raise ValueError("train embedding 数量不一致")
    if load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding 顺序不一致")

    teacher_rows = [
        json.loads(line)
        for line in Path(args.teacher_details).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    teacher_rows.sort(key=lambda row: int(row["index"]))
    if not teacher_rows:
        raise ValueError("teacher details 为空")

    groups = _group_ids(train)
    fit_pool_by_index: dict[int, np.ndarray] = {}
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    for fit_idx, val_idx in folds.split(np.zeros(len(train)), labels, groups):
        for index in val_idx.tolist():
            fit_pool_by_index[index] = fit_idx

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    descriptions = candidate_descriptions(schema, scope="company")
    normalized_vectors = _normalize_rows(vectors)
    selected = [
        row
        for row in teacher_rows
        if row.get("valid")
        and not row.get("abstain")
        and float(row.get("confidence", 0.0)) >= args.min_teacher_confidence
        and row.get("teacher_event")
        and row.get("teacher_event") != row.get("baseline_event")
    ]

    jobs = []
    for row in selected:
        index = int(row["index"])
        names = [str(row["baseline_event"]), str(row["teacher_event"])]
        exemplars = _retrieved_exemplars(
            query_index=index,
            candidate_names=names,
            fit_indices=fit_pool_by_index[index],
            normalized_vectors=normalized_vectors,
            labels=labels,
            articles=train,
            content_chars=args.exemplar_content_chars,
        )
        jobs.append((row, exemplars))

    def run_one(row: dict, exemplars: dict) -> dict:
        index = int(row["index"])
        baseline = str(row["baseline_event"])
        proposed = str(row["teacher_event"])
        client = build_client()
        verifier = CandidateChangeVerifier(client, max_content_chars=4000)
        decision = verifier.verify(
            train[index],
            baseline_event=baseline,
            proposed_event=proposed,
            teacher_reason=str(row.get("reason") or ""),
            candidate_definitions={
                baseline: descriptions.get(baseline, ""),
                proposed: descriptions.get(proposed, ""),
            },
            exemplars=exemplars,
        )
        return {
            "index": index,
            "article_id": row["article_id"],
            "truth": row["truth"],
            "baseline_event": baseline,
            "proposed_event": proposed,
            "teacher_confidence": row["confidence"],
            "accept": decision.accept if decision.valid else False,
            "verifier_confidence": decision.confidence,
            "valid": decision.valid,
            "reason": decision.reason,
            "error": decision.error,
            "latency_ms": decision.latency_ms,
            "usage": dict(client.usage),
        }

    verifier_rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one, *job) for job in jobs]
        for future in as_completed(futures):
            verifier_rows.append(future.result())
    verifier_rows.sort(key=lambda row: row["index"])

    verifier_by_index = {row["index"]: row for row in verifier_rows}
    final_rows = []
    for row in teacher_rows:
        index = int(row["index"])
        verifier_row = verifier_by_index.get(index)
        final_event = row["baseline_event"]
        if verifier_row and verifier_row["accept"]:
            final_event = verifier_row["proposed_event"]
        final_rows.append({**row, "final_event": final_event})

    corrected = sum(
        row["baseline_event"] != row["truth"] and row["final_event"] == row["truth"]
        for row in final_rows
    )
    harmed = sum(
        row["baseline_event"] == row["truth"] and row["final_event"] != row["truth"]
        for row in final_rows
    )
    accepted = [row for row in verifier_rows if row["accept"]]
    accepted_correct = sum(row["proposed_event"] == row["truth"] for row in accepted)
    usage = Counter()
    for row in verifier_rows:
        usage.update(row["usage"])
    latencies = [row["latency_ms"] for row in verifier_rows]
    payload = {
        "scope": "company",
        "protocol": (
            "second-stage conservative verifier over retrieval-augmented Teacher changes only; "
            "same duplication-safe Gold exemplars; baseline is preserved unless verifier accepts"
        ),
        "teacher_sample_count": len(teacher_rows),
        "min_teacher_confidence": args.min_teacher_confidence,
        "reviewed_change_count": len(verifier_rows),
        "accepted_change_count": len(accepted),
        "accepted_change_precision": round(accepted_correct / max(1, len(accepted)), 6),
        "corrected": corrected,
        "harmed": harmed,
        "net_corrections": corrected - harmed,
        "final_accuracy": round(
            sum(row["final_event"] == row["truth"] for row in final_rows) / len(final_rows), 6
        ),
        "invalid_verifier_count": sum(not row["valid"] for row in verifier_rows),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        },
        "api_usage": dict(usage),
    }
    Path(args.details_output).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in verifier_rows) + "\n",
        encoding="utf-8",
    )
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
