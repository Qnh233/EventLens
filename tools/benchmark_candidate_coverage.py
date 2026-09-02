from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_routed_subject_svc_v2 import routed_text
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)[f"{args.scope}_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)[f"{args.scope}_event"]
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if test_manifest.article_count != len(test):
        raise ValueError("test embedding 数量不一致")
    if load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
        raise ValueError("test embedding 顺序不一致")

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
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
        test,
        test_vectors,
        scope=args.scope,
        settings=settings,
        schema=schema,
        client=client,
    )

    labels = [str(row.event_label) for row in train]
    truth = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    description_texts, description_labels = label_description_examples(schema, scope=args.scope)
    train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]
    model = fit_with_descriptions(
        train_texts,
        labels,
        description_texts,
        description_labels,
        repeat=1,
    )
    scores = decision_scores(model, test_texts, classes)
    order = np.argsort(-scores, axis=1)

    rows = {}
    for svc_k in (1, 3, 5, 8):
        for bge_k in (0, 3, 5):
            hit = 0
            sizes = []
            for index, label in enumerate(truth):
                candidates = {classes[j] for j in order[index, :svc_k]}
                if bge_k:
                    candidates.update(row.event_name for row in recalls[index].candidates[:bge_k])
                sizes.append(len(candidates))
                hit += label in candidates
            key = f"svc{svc_k}_bge{bge_k}"
            rows[key] = {
                "hit_rate": round(hit / len(test), 6),
                "hit_count": hit,
                "mean_candidate_count": round(float(np.mean(sizes)), 3),
                "max_candidate_count": max(sizes),
            }

    payload = {
        "scope": args.scope,
        "protocol": "production-like candidate coverage; no true subject fields in classifier text",
        "sample_count": len(test),
        "coverage": rows,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
