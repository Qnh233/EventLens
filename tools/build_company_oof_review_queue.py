from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_pretrained_reranker_cascade import build_candidate_sets, constrained
from benchmark_review_acquisition import _balanced_round_robin
from benchmark_routed_subject_svc_v2 import metric, routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.review_queue import build_low_margin_review_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument(
        "--strategy",
        choices=[
            "low_margin",
            "disagreement_then_margin",
            "disagreement_class_balanced_margin",
        ],
        default="low_margin",
    )
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")

    settings = load_settings()
    articles = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in articles]
    classes = sorted(set(labels))
    texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in articles]

    manifest, vectors = load_exported_vectors(args.embeddings_dir)
    if manifest.article_count != len(articles):
        raise ValueError("embedding count mismatch")
    if load_exported_article_ids(args.embeddings_dir) != [row.article_id for row in articles]:
        raise ValueError("embedding order mismatch")

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    native = settings.native_embedding
    client = NativeSentenceTransformerEmbeddingClient(
        model=native.model,
        device=native.device,
        batch_size=native.batch_size,
        normalize_embeddings=native.normalize_embeddings,
        cache_folder=native.cache_folder,
        local_files_only=True,
    )
    routes, recalls = build_routes_and_recalls(
        articles, vectors, scope="company", settings=settings, schema=schema, client=client
    )

    scores = np.full((len(articles), len(classes)), -1e9, dtype=np.float64)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    for fit_idx, val_idx in folds.split(texts, labels):
        model = fit_with_descriptions(
            [texts[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        scores[val_idx] = decision_scores(model, [texts[i] for i in val_idx], classes)

    predictions = constrained(classes, scores, routes, recalls, schema)
    order = np.argsort(-scores, axis=1)
    margins = scores[np.arange(len(scores)), order[:, 0]] - scores[np.arange(len(scores)), order[:, 1]]
    candidates = build_candidate_sets(scores, classes, recalls, svc_k=5, bge_k=5)
    selected_count = max(1, int(round(len(articles) * args.fraction)))
    if args.strategy == "low_margin":
        selected_indices = np.argsort(margins, kind="stable")[:selected_count].tolist()
        reason = "low_margin_top20pct"
    else:
        svc_top1 = [classes[int(row[0])] for row in order]
        bge_top1 = [row.candidates[0].event_name if row.candidates else "" for row in recalls]
        disagreement = np.asarray(
            [int(left != right) for left, right in zip(svc_top1, bge_top1)], dtype=np.int64
        )
        if args.strategy == "disagreement_class_balanced_margin":
            selected_indices = _balanced_round_robin(
                np.flatnonzero(disagreement == 1), svc_top1, margins, selected_count
            ).tolist()
            reason = "svc_bge_disagreement_class_balanced_then_low_margin"
        else:
            selected_indices = np.lexsort(
                (np.arange(len(articles)), margins, -disagreement)
            )[:selected_count].tolist()
            reason = "svc_bge_disagreement_then_low_margin"
    route_statuses = [
        "hard_route" if route.accepted_subject_code else "candidate_only" for route in routes
    ]
    rows = build_low_margin_review_items(
        article_ids=[row.article_id for row in articles],
        scope="company",
        baseline_events=predictions,
        margins=margins.tolist(),
        candidate_events=candidates,
        route_statuses=route_statuses,
        selected_indices=selected_indices,
        reason=reason,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")

    errors = sum(labels[i] != predictions[i] for i in selected_indices)
    oracle_predictions = list(predictions)
    for i in selected_indices:
        oracle_predictions[i] = labels[i]
    oracle = metric(labels, oracle_predictions)
    report = {
        "scope": "company",
        "protocol": (
            "train OOF review queue; acquisition uses only inference-visible signals; "
            "queue contains no Gold truth; Gold used only for aggregate offline capacity measurement"
        ),
        "baseline_oof": metric(labels, predictions),
        "strategy": args.strategy,
        "fraction": args.fraction,
        "selected_count": len(rows),
        "selected_error_count_offline_only": int(errors),
        "selected_error_rate_offline_only": round(errors / max(1, len(rows)), 6),
        "candidate_truth_hit_rate_offline_only": round(
            sum(labels[i] in candidates[i] for i in selected_indices) / max(1, len(rows)), 6
        ),
        "oracle_macro_f1_offline_only": oracle["macro_f1"],
        "oracle_macro_f1_gain_offline_only": round(
            oracle["macro_f1"] - metric(labels, predictions)["macro_f1"], 6
        ),
        "queue_output": str(output),
        "requires_human_approval": True,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
