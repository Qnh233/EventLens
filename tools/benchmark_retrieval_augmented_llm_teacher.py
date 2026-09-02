from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_llm_teacher_gold import build_client, candidate_descriptions, threshold_metrics
from benchmark_pretrained_reranker_cascade import constrained
from benchmark_routed_subject_svc_v2 import routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.env import load_env_file
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.llm_teacher import CandidateEventTeacher


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _group_ids(articles) -> list[str]:
    return [
        str(row.duplication_id).strip()
        if row.duplication_id not in (None, "")
        else f"article:{row.article_id}"
        for row in articles
    ]


def _retrieved_exemplars(
    *,
    query_index: int,
    candidate_names: list[str],
    fit_indices: np.ndarray,
    normalized_vectors: np.ndarray,
    labels: list[str],
    articles,
    content_chars: int,
) -> dict[str, list[dict[str, str]]]:
    query = normalized_vectors[query_index]
    output: dict[str, list[dict[str, str]]] = {}
    fit_list = fit_indices.tolist()
    for label in candidate_names:
        pool = [index for index in fit_list if labels[index] == label]
        if not pool:
            continue
        similarities = normalized_vectors[pool] @ query
        best = pool[int(np.argmax(similarities))]
        row = articles[best]
        output[label] = [
            {
                "title": row.title,
                "source": row.source,
                "content": (row.content or "")[:content_chars],
                "similarity": round(float(np.max(similarities)), 4),
            }
        ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exemplar-content-chars", type=int, default=500)
    parser.add_argument(
        "--resume-details",
        default=None,
        help="optional prior JSONL; valid rows are reused and only invalid/missing rows are retried",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--details-output", required=True)
    args = parser.parse_args()

    load_env_file()
    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    manifest, vectors = load_exported_vectors(args.train_embeddings_dir)
    if manifest.article_count != len(train):
        raise ValueError("train embedding 数量不一致")
    if load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding 顺序不一致")

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    native = settings.native_embedding
    embedding_client = NativeSentenceTransformerEmbeddingClient(
        model=native.model,
        device=native.device,
        batch_size=native.batch_size,
        normalize_embeddings=native.normalize_embeddings,
        cache_folder=native.cache_folder,
        local_files_only=True,
    )
    routes, recalls = build_routes_and_recalls(
        train,
        vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=embedding_client,
    )

    labels = [str(row.event_label) for row in train]
    classes = sorted(set(labels))
    texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    groups = _group_ids(train)
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    oof_scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    fit_pool_by_index: dict[int, np.ndarray] = {}
    for fit_idx, val_idx in folds.split(texts, labels, groups):
        if set(np.asarray(groups)[fit_idx]) & set(np.asarray(groups)[val_idx]):
            raise AssertionError("duplication group leaked across folds")
        model = fit_with_descriptions(
            [texts[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        oof_scores[val_idx] = decision_scores(model, [texts[i] for i in val_idx], classes)
        for index in val_idx.tolist():
            fit_pool_by_index[index] = fit_idx

    baseline_predictions = constrained(classes, oof_scores, routes, recalls, schema)
    order = np.argsort(-oof_scores, axis=1)
    margins = oof_scores[np.arange(len(train)), order[:, 0]] - oof_scores[np.arange(len(train)), order[:, 1]]
    eligible = [
        i
        for i in range(len(train))
        if recalls[i].candidates
        and classes[int(order[i, 0])] != recalls[i].candidates[0].event_name
    ]
    eligible.sort(key=lambda i: (float(margins[i]), train[i].article_id))
    selected_indices = eligible[: args.sample_count]

    descriptions = candidate_descriptions(schema, scope="company")
    normalized_vectors = _normalize_rows(vectors)
    jobs = []
    candidate_truth_hit = 0
    for index in selected_indices:
        names = [classes[j] for j in order[index, :5]]
        names.extend(row.event_name for row in recalls[index].candidates[:5])
        names = list(dict.fromkeys(names))
        candidate_truth_hit += labels[index] in names
        candidates = [
            {"event_name": name, "description": descriptions.get(name, "")}
            for name in names
        ]
        exemplars = _retrieved_exemplars(
            query_index=index,
            candidate_names=names,
            fit_indices=fit_pool_by_index[index],
            normalized_vectors=normalized_vectors,
            labels=labels,
            articles=train,
            content_chars=args.exemplar_content_chars,
        )
        jobs.append((index, candidates, exemplars))

    previous_by_index: dict[int, dict] = {}
    if args.resume_details:
        resume_path = Path(args.resume_details)
        if resume_path.exists():
            for line in resume_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    previous_by_index[int(row["index"])] = row

    def run_one(index: int, candidates, exemplars) -> dict:
        client = build_client()
        teacher = CandidateEventTeacher(client, max_content_chars=4000)
        decision = teacher.label(train[index], candidates, exemplars=exemplars)
        prior = previous_by_index.get(index)
        usage = Counter(prior.get("usage", {}) if prior else {})
        usage.update(client.usage)
        return {
            "index": index,
            "article_id": train[index].article_id,
            "truth": labels[index],
            "baseline_event": baseline_predictions[index],
            "raw_svc_event": classes[int(order[index, 0])],
            "baseline_margin": round(float(margins[index]), 6),
            "candidate_count": len(candidates),
            "exemplar_label_count": len(exemplars),
            "candidate_truth_hit": labels[index] in {row["event_name"] for row in candidates},
            "teacher_event": decision.event_label,
            "confidence": decision.confidence,
            "abstain": decision.abstain,
            "valid": decision.valid,
            "reason": decision.reason,
            "latency_ms": decision.latency_ms,
            "error": decision.error,
            "attempts": int(prior.get("attempts", 1) if prior else 0) + 1,
            "usage": dict(usage),
        }

    details = [
        row
        for index, row in previous_by_index.items()
        if index in selected_indices and row.get("valid", False)
    ]
    jobs_to_run = [
        job
        for job in jobs
        if job[0] not in previous_by_index or not previous_by_index[job[0]].get("valid", False)
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one, *job) for job in jobs_to_run]
        for future in as_completed(futures):
            details.append(future.result())
    details.sort(key=lambda row: row["index"])

    baseline_correct = sum(row["baseline_event"] == row["truth"] for row in details)
    raw_non_abstain = [row for row in details if row["valid"] and not row["abstain"]]
    raw_correct = sum(row["teacher_event"] == row["truth"] for row in raw_non_abstain)
    usage = Counter()
    for row in details:
        usage.update(row["usage"])
    latencies = [row["latency_ms"] for row in details]
    payload = {
        "scope": "company",
        "protocol": (
            "duplication-safe train OOF hard-case replay; SVC Top5 union BGE Top5 candidates; "
            "each candidate receives one nearest Gold exemplar from that sample's fit fold only; "
            "no labeled-only subject fields in prompt"
        ),
        "sample_count": len(details),
        "candidate_truth_hit_rate": round(candidate_truth_hit / max(1, len(details)), 6),
        "baseline_accuracy": round(baseline_correct / max(1, len(details)), 6),
        "teacher_raw_non_abstain_count": len(raw_non_abstain),
        "teacher_raw_precision": round(raw_correct / max(1, len(raw_non_abstain)), 6),
        "teacher_abstain_count": sum(row["abstain"] for row in details),
        "invalid_count": sum(not row["valid"] for row in details),
        "retried_count": sum(int(row.get("attempts", 1)) > 1 for row in details),
        "thresholds": {
            str(threshold): threshold_metrics(details, threshold)
            for threshold in (0.5, 0.6, 0.7, 0.8, 0.9)
        },
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        },
        "api_usage": dict(usage),
    }
    Path(args.details_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.details_output).open("w", encoding="utf-8") as file:
        for row in details:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
