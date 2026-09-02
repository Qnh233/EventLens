from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from benchmark_long_tail_svc import aligned_scores, build_vectorizer, class_weights
from benchmark_routed_subject_svc_v2 import predict_with_policy, routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    label_description_examples,
)
from eventlens.challenge_evaluation import evaluate_challenge_slices
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.oof_error_analysis import bootstrap_classification_ci


FIXED_C = 1.0
FIXED_CLASS_WEIGHT_POWER = 1.0
FIXED_CONTENT_CHARS = 2400
FIXED_POLICY = "exact_fallback_k5"
FIXED_SEEDS = (17, 29, 42, 73, 101)


def metrics(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _check_embeddings(directory: str, rows) -> np.ndarray:
    manifest, vectors = load_exported_vectors(directory)
    if manifest.article_count != len(rows):
        raise ValueError(f"embedding count mismatch: {directory}")
    if load_exported_article_ids(directory) != [row.article_id for row in rows]:
        raise ValueError(f"embedding order mismatch: {directory}")
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    train_vectors = _check_embeddings(args.train_embeddings_dir, train)
    test_vectors = _check_embeddings(args.test_embeddings_dir, test)

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
        train, train_vectors, scope="company", settings=settings, schema=schema, client=client
    )
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope="company", settings=settings, schema=schema, client=client
    )

    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    train_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=FIXED_CONTENT_CHARS)
        for row in train
    ]
    test_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=FIXED_CONTENT_CHARS)
        for row in test
    ]
    description_texts, description_labels = label_description_examples(schema, scope="company")

    seed_rows: list[dict] = []
    for seed in FIXED_SEEDS:
        scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for fit_idx, val_idx in folds.split(train_texts, labels):
            fold_texts = [train_texts[i] for i in fit_idx] + description_texts
            fold_labels = [labels[i] for i in fit_idx] + description_labels
            vectorizer = build_vectorizer()
            x_fit = vectorizer.fit_transform(fold_texts)
            x_val = vectorizer.transform([train_texts[i] for i in val_idx])

            from sklearn.svm import LinearSVC

            model = LinearSVC(
                C=FIXED_C,
                class_weight=class_weights(fold_labels, power=FIXED_CLASS_WEIGHT_POWER),
                random_state=seed,
            )
            model.fit(x_fit, fold_labels)
            scores[val_idx] = aligned_scores(model, x_val, classes)

        pred = predict_with_policy(
            classes,
            scores,
            train_routes,
            train_recalls,
            schema,
            scope="company",
            policy_name=FIXED_POLICY,
        )
        seed_rows.append({"seed": seed, **metrics(labels, pred)})

    from sklearn.svm import LinearSVC

    fit_texts = train_texts + description_texts
    fit_labels = labels + description_labels
    vectorizer = build_vectorizer()
    fit_started = time.perf_counter()
    x_fit = vectorizer.fit_transform(fit_texts)
    model = LinearSVC(
        C=FIXED_C,
        class_weight=class_weights(fit_labels, power=FIXED_CLASS_WEIGHT_POWER),
        random_state=settings.model.random_state,
    )
    model.fit(x_fit, fit_labels)
    fit_seconds = time.perf_counter() - fit_started

    predict_started = time.perf_counter()
    x_test = vectorizer.transform(test_texts)
    test_scores = aligned_scores(model, x_test, classes)
    test_pred = predict_with_policy(
        classes,
        test_scores,
        test_routes,
        test_recalls,
        schema,
        scope="company",
        policy_name=FIXED_POLICY,
    )
    predict_seconds = time.perf_counter() - predict_started
    external = metrics(test_labels, test_pred)

    challenge_cfg = settings.challenge_evaluation
    challenge = evaluate_challenge_slices(
        train,
        test,
        test_pred,
        test_routes,
        scope="company",
        rare_event_max_train_count=challenge_cfg.rare_event_max_train_count,
        long_tail_source_max_train_count=challenge_cfg.long_tail_source_max_train_count,
        long_text_percentile=challenge_cfg.long_text_percentile,
    )

    payload = {
        "scope": "company",
        "protocol": (
            "fixed production-like no-subject char-SVC recipe; five 3-fold OOF seeds measure "
            "stability only; external tagged test is evaluated once with frozen C/class-weight/text/policy"
        ),
        "fixed_recipe": {
            "C": FIXED_C,
            "class_weight_power": FIXED_CLASS_WEIGHT_POWER,
            "content_chars": FIXED_CONTENT_CHARS,
            "schema_description_repeat": 1,
            "policy": FIXED_POLICY,
            "uses_labeled_only_subject_fields": False,
        },
        "oof_by_seed": seed_rows,
        "oof_macro_f1_summary": summarize([row["macro_f1"] for row in seed_rows]),
        "external": external,
        "external_bootstrap_ci": bootstrap_classification_ci(
            test_labels,
            test_pred,
            n_bootstrap=2000,
            confidence=0.95,
            seed=settings.model.random_state,
        ),
        "challenge_slices": {name: row.model_dump() for name, row in challenge.items()},
        "runtime": {
            "fit_seconds": round(fit_seconds, 6),
            "test_batch_articles": len(test),
            "test_transform_predict_seconds": round(predict_seconds, 6),
            "test_transform_predict_ms_per_article": round(1000.0 * predict_seconds / len(test), 6),
            "classifier_gpu_required": False,
            "classifier_external_api_calls": 0,
            "classifier_external_api_cost": 0.0,
        },
        "production_like_reference_macro_f1": 0.770798,
        "matches_reference": abs(external["macro_f1"] - 0.770798) < 1e-6,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
