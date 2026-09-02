from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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
from eventlens.event_candidate_reranker import build_candidate_feature_rows, choose_best_candidate
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.schema_constrained_classifier import constrain_predictions


def metrics(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def flatten_features(grouped_rows, article_indices):
    features = []
    targets = []
    owners = []
    names = []
    for article_index in article_indices:
        truth = grouped_rows[article_index][0]
        rows = grouped_rows[article_index][1]
        for row in rows:
            features.append(row.features)
            targets.append(int(row.event_name == truth))
            owners.append(article_index)
            names.append(row.event_name)
    return features, targets, owners, names


def fit_pair_model(features, targets, *, c: float, class_weight: str | None):
    # 特征仅几十维、候选对约万级，dense 更简单且规避 sklearn
    # 对 int64 sparse index 的版本兼容限制。
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(features)
    model = LogisticRegression(
        C=c,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    model.fit(matrix, targets)
    return vectorizer, model


def predict_grouped(vectorizer, model, grouped_rows, article_indices):
    output = []
    for article_index in article_indices:
        rows = grouped_rows[article_index][1]
        names = [row.event_name for row in rows]
        matrix = vectorizer.transform([row.features for row in rows])
        probabilities = model.predict_proba(matrix)[:, 1].tolist()
        output.append(choose_best_candidate(names, probabilities))
    return output


def candidate_hit_rate(grouped_rows) -> float:
    hit = sum(
        truth in {row.event_name for row in rows}
        for truth, rows in grouped_rows
    )
    return round(hit / max(1, len(grouped_rows)), 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)[f"{args.scope}_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)[f"{args.scope}_event"]
    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding 顺序不一致")
    if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
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
    train_routes, train_recalls = build_routes_and_recalls(
        train, train_vectors, scope=args.scope, settings=settings, schema=schema, client=client
    )
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope=args.scope, settings=settings, schema=schema, client=client
    )

    train_labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(train_labels))
    description_texts, description_labels = label_description_examples(schema, scope=args.scope)
    train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]

    base_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    oof_scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    for fit_idx, val_idx in base_folds.split(train_texts, train_labels):
        svc = fit_with_descriptions(
            [train_texts[index] for index in fit_idx],
            [train_labels[index] for index in fit_idx],
            description_texts,
            description_labels,
            repeat=1,
        )
        oof_scores[val_idx] = decision_scores(
            svc, [train_texts[index] for index in val_idx], classes
        )

    final_svc = fit_with_descriptions(
        train_texts,
        train_labels,
        description_texts,
        description_labels,
        repeat=1,
    )
    test_scores = decision_scores(final_svc, test_texts, classes)

    train_grouped = [
        (
            train_labels[index],
            build_candidate_feature_rows(
                classes,
                oof_scores[index],
                train_recalls[index],
                train_routes[index],
                schema,
                scope=args.scope,
                svc_top_k=5,
                bge_top_k=5,
                label_counts=None,
            ),
        )
        for index in range(len(train))
    ]
    test_grouped = [
        (
            test_labels[index],
            build_candidate_feature_rows(
                classes,
                test_scores[index],
                test_recalls[index],
                test_routes[index],
                schema,
                scope=args.scope,
                svc_top_k=5,
                bge_top_k=5,
                label_counts=None,
            ),
        )
        for index in range(len(test))
    ]

    meta_folds = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=settings.model.random_state + 17
    )
    candidates = [
        (c, class_weight)
        for c in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
        for class_weight in (None, "balanced")
    ]
    cv_results = {}
    best_key = ""
    best_score = -1.0
    best_spec = candidates[0]
    article_indices = np.arange(len(train))
    for c, class_weight in candidates:
        oof_pred = [""] * len(train)
        for fit_idx, val_idx in meta_folds.split(article_indices, train_labels):
            features, targets, _, _ = flatten_features(train_grouped, fit_idx.tolist())
            vectorizer, model = fit_pair_model(
                features, targets, c=c, class_weight=class_weight
            )
            pred = predict_grouped(vectorizer, model, train_grouped, val_idx.tolist())
            for index, value in zip(val_idx, pred):
                oof_pred[int(index)] = value
        score = metrics(train_labels, oof_pred)
        key = f"C={c}:class_weight={class_weight or 'none'}"
        cv_results[key] = score
        if score["macro_f1"] > best_score:
            best_key = key
            best_score = score["macro_f1"]
            best_spec = (c, class_weight)

    train_features, train_targets, _, _ = flatten_features(
        train_grouped, list(range(len(train)))
    )
    vectorizer, reranker = fit_pair_model(
        train_features,
        train_targets,
        c=best_spec[0],
        class_weight=best_spec[1],
    )
    reranked_pred = predict_grouped(
        vectorizer, reranker, test_grouped, list(range(len(test)))
    )

    global_pred = [classes[int(index)] for index in np.argmax(test_scores, axis=1)]
    exact_fallback_pred = constrain_predictions(
        classes,
        test_scores,
        test_routes,
        test_recalls,
        schema,
        scope=args.scope,
        policy="exact_subject_recall_fallback",
        recall_top_k=5,
    )
    payload = {
        "scope": args.scope,
        "protocol": "production-like two-level OOF candidate reranker; no true subject fields in classifier/reranker features; SVC Top5 union BGE Top5",
        "train_count": len(train),
        "external_count": len(test),
        "train_candidate_hit_rate": candidate_hit_rate(train_grouped),
        "external_candidate_hit_rate": candidate_hit_rate(test_grouped),
        "selected_key": best_key,
        "selected_oof": cv_results[best_key],
        "cv": dict(sorted(cv_results.items(), key=lambda item: item[1]["macro_f1"], reverse=True)),
        "external": {
            "svc_global": metrics(test_labels, global_pred),
            "svc_exact_fallback_k5": metrics(test_labels, exact_fallback_pred),
            "candidate_reranker": metrics(test_labels, reranked_pred),
        },
        "gate_macro_f1": 0.80,
        "gate_passed": f1_score(test_labels, reranked_pred, average="macro", zero_division=0) >= 0.80,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
