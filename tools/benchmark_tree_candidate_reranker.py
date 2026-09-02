from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_routed_subject_svc_v2 import predict_with_policy, routed_text
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


OOF_GAIN_GATE = 0.005
MODEL_SPECS = (
    {"max_leaf_nodes": 15, "max_iter": 120, "learning_rate": 0.08, "l2_regularization": 1.0, "positive_weight": 5.0},
    {"max_leaf_nodes": 31, "max_iter": 160, "learning_rate": 0.06, "l2_regularization": 2.0, "positive_weight": 8.0},
)


def metric(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def duplication_groups(articles) -> list[str]:
    return [
        str(row.duplication_id).strip()
        if row.duplication_id not in (None, "")
        else f"article:{row.article_id}"
        for row in articles
    ]


def flatten(grouped_rows, article_indices):
    features: list[dict] = []
    targets: list[int] = []
    for article_index in article_indices:
        truth, rows = grouped_rows[int(article_index)]
        for row in rows:
            features.append(row.features)
            targets.append(int(row.event_name == truth))
    return features, np.asarray(targets, dtype=np.int64)


def fit_model(features, targets: np.ndarray, spec: dict):
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(features).astype(np.float32, copy=False)
    model = HistGradientBoostingClassifier(
        learning_rate=spec["learning_rate"],
        max_iter=spec["max_iter"],
        max_leaf_nodes=spec["max_leaf_nodes"],
        l2_regularization=spec["l2_regularization"],
        early_stopping=False,
        random_state=42,
    )
    sample_weight = np.where(targets == 1, spec["positive_weight"], 1.0)
    model.fit(matrix, targets, sample_weight=sample_weight)
    return vectorizer, model


def predict_groups(vectorizer, model, grouped_rows, article_indices) -> list[str]:
    output: list[str] = []
    for article_index in article_indices:
        _, rows = grouped_rows[int(article_index)]
        names = [row.event_name for row in rows]
        matrix = vectorizer.transform([row.features for row in rows]).astype(np.float32, copy=False)
        probability = model.predict_proba(matrix)[:, 1].tolist()
        output.append(choose_best_candidate(names, probability))
    return output


def build_grouped(articles, labels, scores, routes, recalls, schema, classes):
    counts = {label: labels.count(label) for label in classes}
    return [
        (
            labels[index],
            build_candidate_feature_rows(
                classes,
                scores[index],
                recalls[index],
                routes[index],
                schema,
                scope="company",
                svc_top_k=5,
                bge_top_k=5,
                label_counts=counts,
            ),
        )
        for index in range(len(articles))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in train]
    classes = sorted(set(labels))
    groups = duplication_groups(train)
    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    if train_manifest.article_count != len(train):
        raise ValueError("train embedding count mismatch")
    if load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding order mismatch")

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
        train,
        train_vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=client,
    )
    description_texts, description_labels = label_description_examples(schema, scope="company")
    texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]

    base_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof_scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    leakage_checks: list[dict[str, int]] = []
    for fold, (fit_idx, val_idx) in enumerate(base_splitter.split(texts, labels, groups), start=1):
        fit_groups = {groups[int(index)] for index in fit_idx}
        val_groups = {groups[int(index)] for index in val_idx}
        overlap = fit_groups & val_groups
        if overlap:
            raise RuntimeError("duplication group leaked across base OOF fold")
        leakage_checks.append({"fold": fold, "overlap_groups": len(overlap)})
        svc = fit_with_descriptions(
            [texts[int(index)] for index in fit_idx],
            [labels[int(index)] for index in fit_idx],
            description_texts,
            description_labels,
            repeat=1,
        )
        oof_scores[val_idx] = decision_scores(
            svc, [texts[int(index)] for index in val_idx], classes
        )

    baseline_pred = predict_with_policy(
        classes,
        oof_scores,
        routes,
        recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    baseline = metric(labels, baseline_pred)
    grouped = build_grouped(train, labels, oof_scores, routes, recalls, schema, classes)

    meta_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=91)
    indices = np.arange(len(train))
    cv: dict[str, dict[str, float]] = {}
    best_key = ""
    best_spec = MODEL_SPECS[0]
    best_macro = -1.0
    for spec in MODEL_SPECS:
        pred = [""] * len(train)
        for fit_idx, val_idx in meta_splitter.split(indices, labels, groups):
            fit_groups = {groups[int(index)] for index in fit_idx}
            val_groups = {groups[int(index)] for index in val_idx}
            if fit_groups & val_groups:
                raise RuntimeError("duplication group leaked across meta OOF fold")
            features, targets = flatten(grouped, fit_idx)
            vectorizer, model = fit_model(features, targets, spec)
            fold_pred = predict_groups(vectorizer, model, grouped, val_idx)
            for index, value in zip(val_idx, fold_pred):
                pred[int(index)] = value
        score = metric(labels, pred)
        key = (
            f"leaf={spec['max_leaf_nodes']}:iter={spec['max_iter']}:"
            f"lr={spec['learning_rate']}:l2={spec['l2_regularization']}:pw={spec['positive_weight']}"
        )
        cv[key] = score
        if score["macro_f1"] > best_macro:
            best_key = key
            best_macro = score["macro_f1"]
            best_spec = dict(spec)

    gain = round(best_macro - baseline["macro_f1"], 6)
    payload: dict[str, object] = {
        "scope": "company",
        "protocol": (
            "production-like duplication-safe two-level OOF nonlinear candidate reranker; "
            "features use only inference-visible SVC/BGE/schema-route signals; external locked unless OOF gain >=0.005"
        ),
        "leakage_checks": leakage_checks,
        "baseline_group_safe_oof": baseline,
        "selected_key": best_key,
        "selected_spec": best_spec,
        "selected_oof": cv[best_key],
        "oof_gain": gain,
        "cv": cv,
        "external_gate_gain": OOF_GAIN_GATE,
        "external_touched": False,
        "production_like_external_reference_macro_f1": 0.772131,
    }

    if gain >= OOF_GAIN_GATE:
        test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
        test_labels = [str(row.event_label) for row in test]
        test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
        if test_manifest.article_count != len(test):
            raise ValueError("test embedding count mismatch")
        if load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
            raise ValueError("test embedding order mismatch")
        test_routes, test_recalls = build_routes_and_recalls(
            test,
            test_vectors,
            scope="company",
            settings=settings,
            schema=schema,
            client=client,
        )
        final_svc = fit_with_descriptions(
            texts,
            labels,
            description_texts,
            description_labels,
            repeat=1,
        )
        test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]
        test_scores = decision_scores(final_svc, test_texts, classes)
        test_grouped = build_grouped(
            test, test_labels, test_scores, test_routes, test_recalls, schema, classes
        )
        features, targets = flatten(grouped, indices)
        vectorizer, model = fit_model(features, targets, best_spec)
        test_pred = predict_groups(vectorizer, model, test_grouped, np.arange(len(test)))
        payload["external_touched"] = True
        payload["external"] = metric(test_labels, test_pred)

    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
