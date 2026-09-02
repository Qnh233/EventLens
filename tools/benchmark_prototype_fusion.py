from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_routed_subject_svc_v2 import metric, predict_with_policy, routed_text
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


WEIGHTS = [0.0, 0.1, 0.25, 0.5]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def _normalize_vectors(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _prototype_scores(
    train_vectors: np.ndarray,
    train_labels: list[str],
    query_vectors: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    train = _normalize_vectors(np.asarray(train_vectors, dtype=np.float64))
    query = _normalize_vectors(np.asarray(query_vectors, dtype=np.float64))
    prototypes = []
    for label in classes:
        indices = [i for i, value in enumerate(train_labels) if value == label]
        prototype = train[indices].mean(axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
        prototypes.append(prototype)
    return query @ np.asarray(prototypes, dtype=np.float64).T


def _schema_scores(client, description_texts, description_labels, query_vectors, classes):
    description_vectors = _normalize_vectors(
        np.asarray(client.embed(description_texts), dtype=np.float64)
    )
    by_label = {label: vector for label, vector in zip(description_labels, description_vectors)}
    matrix = np.asarray([by_label[label] for label in classes], dtype=np.float64)
    return _normalize_vectors(np.asarray(query_vectors, dtype=np.float64)) @ matrix.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--text-mode", choices=["no_subject", "route_hard", "route_top3"], required=True)
    parser.add_argument("--max-content-chars", type=int, required=True)
    parser.add_argument("--description-repeat", type=int, choices=[0, 1], required=True)
    parser.add_argument("--constraint-policy", choices=["global", "exact_fallback_k5", "hard_recall_k3"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)[f"{args.scope}_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)[f"{args.scope}_event"]
    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [x.article_id for x in train]:
        raise ValueError("train embedding 顺序不一致")
    if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [x.article_id for x in test]:
        raise ValueError("test embedding 顺序不一致")

    native = settings.native_embedding
    client = NativeSentenceTransformerEmbeddingClient(
        model=native.model,
        device=native.device,
        batch_size=native.batch_size,
        normalize_embeddings=native.normalize_embeddings,
        cache_folder=native.cache_folder,
        local_files_only=True,
    )
    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    train_routes, train_recalls = build_routes_and_recalls(
        train, train_vectors, scope=args.scope, settings=settings, schema=schema, client=client
    )
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope=args.scope, settings=settings, schema=schema, client=client
    )

    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    description_texts, description_labels = label_description_examples(schema, scope=args.scope)
    train_texts = [
        routed_text(row, route, mode=args.text_mode, max_content_chars=args.max_content_chars)
        for row, route in zip(train, train_routes)
    ]
    test_texts = [
        routed_text(row, route, mode=args.text_mode, max_content_chars=args.max_content_chars)
        for row, route in zip(test, test_routes)
    ]

    schema_oof_scores = _schema_scores(client, description_texts, description_labels, train_vectors, classes)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    svc_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    proto_oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    for train_idx, val_idx in folds.split(train_texts, labels):
        model = fit_with_descriptions(
            [train_texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            description_texts,
            description_labels,
            repeat=args.description_repeat,
        )
        svc_oof[val_idx] = decision_scores(model, [train_texts[i] for i in val_idx], classes)
        proto_oof[val_idx] = _prototype_scores(
            np.asarray(train_vectors)[train_idx],
            [labels[i] for i in train_idx],
            np.asarray(train_vectors)[val_idx],
            classes,
        )

    svc_z = _normalize_rows(svc_oof)
    proto_z = _normalize_rows(proto_oof)
    schema_z = _normalize_rows(schema_oof_scores)
    oof = {}
    best = None
    best_score = -1.0
    for proto_weight in WEIGHTS:
        for schema_weight in WEIGHTS:
            fused = svc_z + proto_weight * proto_z + schema_weight * schema_z
            pred = predict_with_policy(
                classes,
                fused,
                train_routes,
                train_recalls,
                schema,
                scope=args.scope,
                policy_name=args.constraint_policy,
            )
            row = metric(labels, pred)
            key = f"proto{proto_weight}:schema{schema_weight}"
            oof[key] = row
            if row["macro_f1"] > best_score:
                best_score = row["macro_f1"]
                best = (proto_weight, schema_weight, key)

    assert best is not None
    proto_weight, schema_weight, best_key = best
    model = fit_with_descriptions(
        train_texts,
        labels,
        description_texts,
        description_labels,
        repeat=args.description_repeat,
    )
    svc_test = decision_scores(model, test_texts, classes)
    proto_test = _prototype_scores(train_vectors, labels, test_vectors, classes)
    schema_test = _schema_scores(client, description_texts, description_labels, test_vectors, classes)
    fused_test = (
        _normalize_rows(svc_test)
        + proto_weight * _normalize_rows(proto_test)
        + schema_weight * _normalize_rows(schema_test)
    )
    test_pred = predict_with_policy(
        classes,
        fused_test,
        test_routes,
        test_recalls,
        schema,
        scope=args.scope,
        policy_name=args.constraint_policy,
    )
    external = metric(test_labels, test_pred)
    payload = {
        "scope": args.scope,
        "protocol": "production-like routed subject + OOF-selected BGE data/schema prototype fusion",
        "base_config": {
            "text_mode": args.text_mode,
            "max_content_chars": args.max_content_chars,
            "description_repeat": args.description_repeat,
            "constraint_policy": args.constraint_policy,
        },
        "selected_weights": {"prototype": proto_weight, "schema": schema_weight},
        "selected_oof": oof[best_key],
        "external": external,
        "gate_macro_f1": 0.80,
        "gate_passed": external["macro_f1"] >= 0.80,
        "oof": dict(sorted(oof.items(), key=lambda item: item[1]["macro_f1"], reverse=True)),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
