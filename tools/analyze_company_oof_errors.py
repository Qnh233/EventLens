from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from benchmark_pretrained_reranker_cascade import build_candidate_sets, constrained
from benchmark_routed_subject_svc_v2 import routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.challenge_evaluation import evaluate_challenge_slices
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.oof_error_analysis import (
    bootstrap_classification_ci,
    candidate_error_coverage,
    confusion_concentration,
    hardcase_oracle_frontier,
    margin_error_profile,
)


def _metrics(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

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
        articles,
        vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=client,
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
    is_error = np.asarray([gold != pred for gold, pred in zip(labels, predictions)], dtype=bool)

    candidates = build_candidate_sets(scores, classes, recalls, svc_k=5, bge_k=5)
    svc_top5 = [[classes[int(j)] for j in row[:5]] for row in order]
    bge_top5 = [[candidate.event_name for candidate in row.candidates[:5]] for row in recalls]

    oracle_pred = list(predictions)
    for index, (gold, pred, group) in enumerate(zip(labels, predictions, candidates)):
        if gold != pred and gold in group:
            oracle_pred[index] = gold

    challenge = settings.challenge_evaluation
    challenge_report = evaluate_challenge_slices(
        articles,
        articles,
        predictions,
        routes,
        scope="company",
        rare_event_max_train_count=challenge.rare_event_max_train_count,
        long_tail_source_max_train_count=challenge.long_tail_source_max_train_count,
        long_text_percentile=challenge.long_text_percentile,
    )

    payload = {
        "scope": "company",
        "protocol": "production-like company train 3-fold OOF error analysis; no labeled-only subject fields used by classifier/reranker; Gold used only for post-hoc evaluation",
        "sample_count": len(articles),
        "baseline_oof": _metrics(labels, predictions),
        "baseline_oof_bootstrap_ci": bootstrap_classification_ci(
            labels,
            predictions,
            n_bootstrap=1000,
            confidence=0.95,
            seed=settings.model.random_state,
        ),
        "oracle_if_union_candidate_is_always_correct": _metrics(labels, oracle_pred),
        "union_top5_error_coverage": candidate_error_coverage(labels, predictions, candidates),
        "svc_top5_error_coverage": candidate_error_coverage(labels, predictions, svc_top5),
        "bge_top5_error_coverage": candidate_error_coverage(labels, predictions, bge_top5),
        "margin_error_profile": margin_error_profile(margins, is_error),
        "hardcase_oracle_frontier": hardcase_oracle_frontier(labels, predictions, margins),
        "confusion_concentration": confusion_concentration(labels, predictions, top_n=15),
        "challenge_slices": {
            name: row.model_dump() for name, row in challenge_report.items()
        },
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
