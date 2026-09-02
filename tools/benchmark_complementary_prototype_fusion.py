from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_retrieval_augmented_llm_teacher import _group_ids
from benchmark_routed_subject_svc_v2 import metric, predict_with_policy, routed_text
from benchmark_company_temporal_holdout import paired_bootstrap_macro_f1_gain
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.exemplar_reranker import topk_exemplar_scores
from eventlens.io import read_competition_labeled_excel
from eventlens.rocchio import build_rocchio_prototypes, rocchio_scores


EXEMPLAR_WEIGHT = 0.1
ROCCHIO_WEIGHTS = (0.1, 0.2)
OOF_GAIN_GATE = 0.005


def _row_zscore(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / np.maximum(std, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Run train-only OOF stability audit and never read tagged test.",
    )
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in train]
    classes = sorted(set(labels))
    groups = _group_ids(train)
    train_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train
    ]
    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    if train_manifest.article_count != len(train):
        raise ValueError("train embedding count mismatch")
    if load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding order mismatch")

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
    train_routes, train_recalls = build_routes_and_recalls(
        train,
        train_vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=client,
    )

    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=settings.model.random_state,
    )
    svc_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    exemplar_oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    rocchio_oof = np.zeros((len(train), len(classes)), dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    leakage_checks = []
    fold_ids = np.zeros(len(train), dtype=np.int64)
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(train_texts, labels, groups), start=1):
        if set(group_array[fit_idx]) & set(group_array[val_idx]):
            raise AssertionError("duplication group leaked across folds")
        leakage_checks.append({"fold": fold, "overlap_groups": 0})
        fold_ids[val_idx] = fold
        model = fit_with_descriptions(
            [train_texts[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        val_texts = [train_texts[i] for i in val_idx]
        svc_oof[val_idx] = decision_scores(model, val_texts, classes)
        exemplar_oof[val_idx] = topk_exemplar_scores(
            np.asarray(train_vectors)[fit_idx],
            [labels[i] for i in fit_idx],
            np.asarray(train_vectors)[val_idx],
            classes,
            top_k=1,
        )
        vectorizer = model.named_steps["tfidf"]
        fit_features = vectorizer.transform([train_texts[i] for i in fit_idx])
        prototypes = build_rocchio_prototypes(
            fit_features,
            [labels[i] for i in fit_idx],
            classes,
            negative_weight=0.0,
        )
        rocchio_oof[val_idx] = rocchio_scores(vectorizer.transform(val_texts), prototypes)

    baseline_pred = predict_with_policy(
        classes,
        svc_oof,
        train_routes,
        train_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    baseline = metric(labels, baseline_pred)
    svc_z = _row_zscore(svc_oof)
    exemplar_z = _row_zscore(exemplar_oof)
    rocchio_z = _row_zscore(rocchio_oof)
    search = {}
    best_weight = 0.0
    best_metric = baseline
    for rocchio_weight in ROCCHIO_WEIGHTS:
        fused = (
            svc_z
            + EXEMPLAR_WEIGHT * exemplar_z
            + rocchio_weight * rocchio_z
        )
        pred = predict_with_policy(
            classes,
            fused,
            train_routes,
            train_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        row = metric(labels, pred)
        search[f"exemplar=0.1:rocchio={rocchio_weight}"] = row
        if row["macro_f1"] > best_metric["macro_f1"]:
            best_metric = row
            best_weight = rocchio_weight

    gain = round(best_metric["macro_f1"] - baseline["macro_f1"], 6)
    selected_fused = (
        svc_z
        + EXEMPLAR_WEIGHT * exemplar_z
        + best_weight * rocchio_z
    )
    selected_pred = predict_with_policy(
        classes,
        selected_fused,
        train_routes,
        train_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    fold_stability = []
    for fold in range(1, 4):
        idx = np.flatnonzero(fold_ids == fold)
        fold_baseline = metric([labels[i] for i in idx], [baseline_pred[i] for i in idx])
        fold_candidate = metric([labels[i] for i in idx], [selected_pred[i] for i in idx])
        fold_stability.append(
            {
                "fold": fold,
                "count": int(len(idx)),
                "baseline_macro_f1": fold_baseline["macro_f1"],
                "candidate_macro_f1": fold_candidate["macro_f1"],
                "gain": round(fold_candidate["macro_f1"] - fold_baseline["macro_f1"], 6),
            }
        )
    payload = {
        "scope": "company",
        "protocol": (
            "production-like duplication-safe 3-fold OOF fixed complementary prototype fusion; "
            "BGE top1 exemplar weight fixed from prior OOF=0.1; TFIDF centroid beta fixed=0; "
            "only two low-degree Rocchio weights compared; external only after +0.005 OOF gate"
        ),
        "leakage_checks": leakage_checks,
        "baseline_group_safe_oof": baseline,
        "search": search,
        "selected": {
            "exemplar_top_k": 1,
            "exemplar_weight": EXEMPLAR_WEIGHT,
            "rocchio_negative_weight": 0.0,
            "rocchio_weight": best_weight,
        },
        "selected_oof": best_metric,
        "oof_gain": gain,
        "oof_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            labels,
            baseline_pred,
            selected_pred,
            n_bootstrap=2000,
            seed=20260830,
        ),
        "fold_stability": fold_stability,
        "all_folds_positive": all(row["gain"] > 0 for row in fold_stability),
        "external_gate_gain": OOF_GAIN_GATE,
        "external_touched": False,
        "production_like_external_reference_macro_f1": 0.770798,
    }

    if not args.skip_external and best_weight > 0.0 and gain >= OOF_GAIN_GATE:
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
        test_texts = [
            routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test
        ]
        model = fit_with_descriptions(train_texts, labels, desc_texts, desc_labels, repeat=1)
        svc_test = decision_scores(model, test_texts, classes)
        exemplar_test = topk_exemplar_scores(
            train_vectors,
            labels,
            test_vectors,
            classes,
            top_k=1,
        )
        vectorizer = model.named_steps["tfidf"]
        prototypes = build_rocchio_prototypes(
            vectorizer.transform(train_texts),
            labels,
            classes,
            negative_weight=0.0,
        )
        rocchio_test = rocchio_scores(vectorizer.transform(test_texts), prototypes)
        fused_test = (
            _row_zscore(svc_test)
            + EXEMPLAR_WEIGHT * _row_zscore(exemplar_test)
            + best_weight * _row_zscore(rocchio_test)
        )
        test_pred = predict_with_policy(
            classes,
            fused_test,
            test_routes,
            test_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        payload["external"] = metric(test_labels, test_pred)
        payload["external_touched"] = True

    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
