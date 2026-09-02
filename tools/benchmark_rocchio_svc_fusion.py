from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_pretrained_reranker_cascade import constrained
from benchmark_retrieval_augmented_llm_teacher import _group_ids
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
from eventlens.rocchio import build_rocchio_prototypes, rocchio_scores
from eventlens.score_fusion import blend_rank_scores


NEGATIVE_WEIGHTS = (0.0, 0.25, 0.5)
FUSION_WEIGHTS = (0.05, 0.1, 0.2, 0.3)


def _fit_fold(
    fit_texts: list[str],
    fit_labels: list[str],
    query_texts: list[str],
    *,
    classes: list[str],
    description_texts: list[str],
    description_labels: list[str],
):
    model = fit_with_descriptions(
        fit_texts,
        fit_labels,
        description_texts,
        description_labels,
        repeat=1,
    )
    svc = decision_scores(model, query_texts, classes)
    vectorizer = model.named_steps["tfidf"]
    fit_features = vectorizer.transform(fit_texts)
    query_features = vectorizer.transform(query_texts)
    rocchio = {}
    for beta in NEGATIVE_WEIGHTS:
        prototypes = build_rocchio_prototypes(
            fit_features,
            fit_labels,
            classes,
            negative_weight=beta,
        )
        rocchio[beta] = rocchio_scores(query_features, prototypes)
    return svc, rocchio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--external-gate-gain", type=float, default=0.005)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
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
    train_routes, train_recalls = build_routes_and_recalls(
        train,
        train_vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=client,
    )

    labels = [str(row.event_label) for row in train]
    classes = sorted(set(labels))
    texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    description_texts, description_labels = label_description_examples(schema, scope="company")
    groups = _group_ids(train)
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    svc_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    rocchio_oof = {
        beta: np.zeros((len(train), len(classes)), dtype=np.float64)
        for beta in NEGATIVE_WEIGHTS
    }
    leakage_checks = []
    group_array = np.asarray(groups, dtype=object)
    for fold, (fit_idx, val_idx) in enumerate(folds.split(texts, labels, groups), start=1):
        overlap = set(group_array[fit_idx]) & set(group_array[val_idx])
        if overlap:
            raise AssertionError("duplication group leaked across folds")
        fold_svc, fold_rocchio = _fit_fold(
            [texts[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            [texts[i] for i in val_idx],
            classes=classes,
            description_texts=description_texts,
            description_labels=description_labels,
        )
        svc_oof[val_idx] = fold_svc
        for beta in NEGATIVE_WEIGHTS:
            rocchio_oof[beta][val_idx] = fold_rocchio[beta]
        leakage_checks.append({"fold": fold, "overlap_groups": 0})

    baseline_pred = constrained(classes, svc_oof, train_routes, train_recalls, schema)
    baseline = metric(labels, baseline_pred)
    search = {}
    best_key = ""
    best_metric = baseline
    best_beta = 0.0
    best_weight = 0.0
    for beta in NEGATIVE_WEIGHTS:
        for weight in FUSION_WEIGHTS:
            fused = blend_rank_scores(svc_oof, rocchio_oof[beta], semantic_weight=weight)
            pred = constrained(classes, fused, train_routes, train_recalls, schema)
            row = metric(labels, pred)
            key = f"beta={beta}:weight={weight}"
            search[key] = row
            if row["macro_f1"] > best_metric["macro_f1"]:
                best_key = key
                best_metric = row
                best_beta = beta
                best_weight = weight

    gain = round(best_metric["macro_f1"] - baseline["macro_f1"], 6)
    payload = {
        "scope": "company",
        "protocol": (
            "production-like duplication-safe 3-fold OOF char-TFIDF Rocchio prototype fusion; "
            "beta/fusion weight selected only on train OOF; external read only if OOF gain passes gate"
        ),
        "leakage_checks": leakage_checks,
        "baseline_group_safe_oof": baseline,
        "search": search,
        "selected_key": best_key or "baseline",
        "selected_oof": best_metric,
        "oof_gain": gain,
        "external_gate_gain": args.external_gate_gain,
        "external_touched": False,
        "production_like_external_reference_macro_f1": 0.770798,
    }

    if gain >= args.external_gate_gain:
        test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
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
        test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]
        model = fit_with_descriptions(
            texts,
            labels,
            description_texts,
            description_labels,
            repeat=1,
        )
        svc_test = decision_scores(model, test_texts, classes)
        vectorizer = model.named_steps["tfidf"]
        prototypes = build_rocchio_prototypes(
            vectorizer.transform(texts),
            labels,
            classes,
            negative_weight=best_beta,
        )
        rocchio_test = rocchio_scores(vectorizer.transform(test_texts), prototypes)
        fused_test = blend_rank_scores(svc_test, rocchio_test, semantic_weight=best_weight)
        test_pred = constrained(classes, fused_test, test_routes, test_recalls, schema)
        payload["external"] = metric([str(row.event_label) for row in test], test_pred)
        payload["external_touched"] = True

    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
