from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_routed_subject_svc_v2 import routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.env import env_float, env_int, env_str, load_env_file
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.llm_agent import OpenAICompatibleChatClient
from eventlens.llm_teacher import CandidateEventTeacher


def candidate_descriptions(schema: EventSchemaIndex, *, scope: str) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for row in schema.definitions:
        if row.scope != scope:
            continue
        values = grouped.setdefault(row.event_name, [])
        text = str(row.description or "").strip()
        if text and text not in values:
            values.append(text)
    return {
        label: "；".join(values[:2])[:900]
        for label, values in grouped.items()
    }


def build_client():
    settings = load_settings()
    return OpenAICompatibleChatClient(
        base_url=env_str("EVENTLENS_LLM_BASE_URL", settings.agent_expert.base_url),
        model=env_str("EVENTLENS_LLM_MODEL", settings.agent_expert.model),
        api_key_env=env_str("EVENTLENS_LLM_API_KEY_ENV", settings.agent_expert.api_key_env),
        temperature=env_float("EVENTLENS_LLM_TEMPERATURE", settings.agent_expert.temperature),
        max_tokens=env_int("EVENTLENS_LLM_MAX_TOKENS", 1024),
        timeout_seconds=env_float("EVENTLENS_LLM_TIMEOUT_SECONDS", 180.0),
        thinking=env_str("EVENTLENS_LLM_THINKING", "enabled"),
        reasoning_effort=env_str("EVENTLENS_LLM_REASONING_EFFORT", "max"),
        json_output=True,
    )


def threshold_metrics(rows: list[dict], threshold: float) -> dict[str, float | int]:
    accepted = [
        row for row in rows
        if row["valid"] and not row["abstain"] and row["confidence"] >= threshold
    ]
    correct = sum(row["teacher_event"] == row["truth"] for row in accepted)
    corrected = sum(
        row["baseline_event"] != row["truth"] and row["teacher_event"] == row["truth"]
        for row in accepted
    )
    harmed = sum(
        row["baseline_event"] == row["truth"] and row["teacher_event"] != row["truth"]
        for row in accepted
    )
    return {
        "accepted_count": len(accepted),
        "precision": round(correct / max(1, len(accepted)), 6),
        "coverage": round(len(accepted) / max(1, len(rows)), 6),
        "corrected": corrected,
        "harmed": harmed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company"], default="company")
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--selection-strategy",
        choices=["long_tail_low_margin", "disagreement_then_margin"],
        default="long_tail_low_margin",
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
    train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    description_texts, description_labels = label_description_examples(schema, scope="company")
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    oof_scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    for fit_idx, val_idx in folds.split(train_texts, labels):
        model = fit_with_descriptions(
            [train_texts[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            description_texts,
            description_labels,
            repeat=1,
        )
        oof_scores[val_idx] = decision_scores(
            model,
            [train_texts[i] for i in val_idx],
            classes,
        )

    order = np.argsort(-oof_scores, axis=1)
    margins = oof_scores[np.arange(len(train)), order[:, 0]] - oof_scores[np.arange(len(train)), order[:, 1]]
    if args.selection_strategy == "disagreement_then_margin":
        eligible = [
            i
            for i in range(len(train))
            if recalls[i].candidates
            and classes[int(order[i, 0])] != recalls[i].candidates[0].event_name
        ]
    else:
        counts = Counter(labels)
        median_count = float(np.median(list(counts.values())))
        long_tail = {label for label, count in counts.items() if count <= median_count}
        eligible = [i for i, label in enumerate(labels) if label in long_tail]
    eligible.sort(key=lambda i: (float(margins[i]), train[i].article_id))
    selected_indices = eligible[: args.sample_count]

    descriptions = candidate_descriptions(schema, scope="company")
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
        jobs.append((index, candidates))

    def run_one(index: int, candidates: list[dict[str, str]]) -> dict:
        client = build_client()
        teacher = CandidateEventTeacher(client, max_content_chars=4000)
        decision = teacher.label(train[index], candidates)
        return {
            "index": index,
            "article_id": train[index].article_id,
            "truth": labels[index],
            "baseline_event": classes[int(order[index, 0])],
            "baseline_margin": round(float(margins[index]), 6),
            "candidate_count": len(candidates),
            "candidate_truth_hit": labels[index] in {row["event_name"] for row in candidates},
            "teacher_event": decision.event_label,
            "confidence": decision.confidence,
            "abstain": decision.abstain,
            "valid": decision.valid,
            "reason": decision.reason,
            "latency_ms": decision.latency_ms,
            "error": decision.error,
            "usage": dict(client.usage),
        }

    details = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one, index, candidates) for index, candidates in jobs]
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
            "train OOF selective Gold replay; no true subject fields in prompt; "
            "SVC Top5 union BGE Top5 candidates; selection uses inference-visible signals only"
        ),
        "selection_strategy": args.selection_strategy,
        "sample_count": len(details),
        "candidate_truth_hit_rate": round(candidate_truth_hit / max(1, len(details)), 6),
        "baseline_accuracy": round(baseline_correct / max(1, len(details)), 6),
        "teacher_raw_non_abstain_count": len(raw_non_abstain),
        "teacher_raw_precision": round(raw_correct / max(1, len(raw_non_abstain)), 6),
        "teacher_abstain_count": sum(row["abstain"] for row in details),
        "invalid_count": sum(not row["valid"] for row in details),
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
