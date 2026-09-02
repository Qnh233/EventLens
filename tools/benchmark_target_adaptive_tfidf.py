from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_routed_subject_svc_v2 import metric, predict_with_policy, routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_articles_excel, read_competition_labeled_excel
from eventlens.target_adaptive_tfidf import decision_scores, fit_target_adaptive_svc


OOF_GAIN_GATE = 0.005


def parse_counts(raw: str) -> list[int]:
    counts = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not counts or counts[0] < 0:
        raise ValueError("unlabeled counts must be non-negative")
    return counts


def duplication_groups(articles) -> list[str]:
    return [
        str(row.duplication_id).strip()
        if row.duplication_id not in (None, "")
        else f"article:{row.article_id}"
        for row in articles
    ]


def sample_domain_texts(articles, count: int, *, seed: int) -> list[str]:
    if count <= 0:
        return []
    if count > len(articles):
        raise ValueError("requested unlabeled count exceeds available articles")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(articles), size=count, replace=False))
    return [
        routed_text(articles[int(index)], None, mode="no_subject", max_content_chars=2400)
        for index in indices
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--unlabeled-counts", default="0,20000,50000")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    counts = parse_counts(args.unlabeled_counts)

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in train]
    classes = sorted(set(labels))
    groups = duplication_groups(train)
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

    # 一次读取无标签目标域数据，所有候选规模复用同一固定随机样本前缀。
    unlabeled = read_articles_excel(
        settings.paths.untagged_train,
        sheet_name=0,
        task_scope="unlabeled_target_domain",
    )
    max_count = max(counts)
    rng = np.random.default_rng(settings.model.random_state)
    sampled_indices = (
        np.sort(rng.choice(len(unlabeled), size=max_count, replace=False))
        if max_count > 0
        else np.empty(0, dtype=np.int64)
    )
    sampled_domain_texts = [
        routed_text(unlabeled[int(index)], None, mode="no_subject", max_content_chars=2400)
        for index in sampled_indices
    ]

    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=settings.model.random_state,
    )
    fold_indices = list(splitter.split(train_texts, labels, groups))
    results: dict[str, dict[str, float]] = {}
    oof_scores_by_count: dict[int, np.ndarray] = {}
    for count in counts:
        domain_texts = sampled_domain_texts[:count]
        scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
        for fit_idx, val_idx in fold_indices:
            model = fit_target_adaptive_svc(
                [train_texts[i] for i in fit_idx],
                [labels[i] for i in fit_idx],
                domain_texts=domain_texts,
                description_texts=desc_texts,
                description_labels=desc_labels,
                random_state=settings.model.random_state,
            )
            scores[val_idx] = decision_scores(
                model, [train_texts[i] for i in val_idx], classes
            )
        pred = predict_with_policy(
            classes,
            scores,
            train_routes,
            train_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        results[str(count)] = metric(labels, pred)
        oof_scores_by_count[count] = scores

    baseline = results[str(min(counts))] if min(counts) == 0 else None
    if baseline is None:
        raise ValueError("unlabeled counts must include 0 for a controlled baseline")
    best_count = max(counts, key=lambda count: results[str(count)]["macro_f1"])
    best_score = results[str(best_count)]["macro_f1"]
    gain = round(best_score - baseline["macro_f1"], 6)
    payload: dict[str, object] = {
        "scope": "company",
        "protocol": (
            "production-like duplication-safe 3-fold OOF target-adaptive TF-IDF; unlabeled target-domain "
            "news may fit char vocabulary/IDF only; LinearSVC supervision uses Gold + one Schema description; "
            "external evaluated only if train OOF gain >= 0.005"
        ),
        "unlabeled_total": len(unlabeled),
        "counts": counts,
        "oof": results,
        "selected_count": best_count,
        "selected_oof_macro_f1": best_score,
        "oof_gain": gain,
        "external_gate_gain": OOF_GAIN_GATE,
        "external_touched": False,
        "production_like_external_reference_macro_f1": 0.770798,
    }

    if best_count > 0 and gain >= OOF_GAIN_GATE:
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
        model = fit_target_adaptive_svc(
            train_texts,
            labels,
            domain_texts=sampled_domain_texts[:best_count],
            description_texts=desc_texts,
            description_labels=desc_labels,
            random_state=settings.model.random_state,
        )
        test_texts = [
            routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test
        ]
        test_scores = decision_scores(model, test_texts, classes)
        test_pred = predict_with_policy(
            classes,
            test_scores,
            test_routes,
            test_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        external = metric(test_labels, test_pred)
        payload.update(
            {
                "external_touched": True,
                "external": external,
                "gain_vs_external_reference": round(
                    external["macro_f1"] - 0.770798, 6
                ),
                "gate_macro_f1": 0.80,
                "stretch_target_macro_f1": 0.85,
                "gate_passed": external["macro_f1"] >= 0.80,
            }
        )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
