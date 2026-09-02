from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from benchmark_pretrained_reranker_cascade import constrained
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


CS = (0.1,)
OOF_GATE_GAIN = 0.005


def nb_log_count_ratio(x: sparse.csr_matrix, positive: np.ndarray) -> np.ndarray:
    """按训练折估计每类对其余类别的平滑 log-count ratio。"""
    pos = np.asarray(positive, dtype=bool)
    neg = ~pos
    p = (np.asarray(x[pos].sum(axis=0)).ravel() + 1.0) / (float(pos.sum()) + 1.0)
    q = (np.asarray(x[neg].sum(axis=0)).ravel() + 1.0) / (float(neg.sum()) + 1.0)
    return np.log(p / q)


def nbsvm_scores(
    fit_texts: list[str],
    fit_labels: list[str],
    eval_texts: list[str],
    classes: list[str],
    *,
    c: float,
) -> np.ndarray:
    vectorizer = CountVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=80000,
        binary=True,
    )
    x_fit = vectorizer.fit_transform(fit_texts).tocsr()
    x_eval = vectorizer.transform(eval_texts).tocsr()
    labels = np.asarray(fit_labels, dtype=object)
    def fit_one(label: str) -> np.ndarray:
        positive = labels == label
        ratio = nb_log_count_ratio(x_fit, positive)
        clf = LinearSVC(C=c, class_weight="balanced", random_state=42)
        clf.fit(x_fit.multiply(ratio), positive.astype(np.int8))
        return np.asarray(clf.decision_function(x_eval.multiply(ratio)), dtype=np.float64)

    columns = Parallel(n_jobs=8, prefer="threads")(
        delayed(fit_one)(label) for label in classes
    )
    return np.column_stack(columns)


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
    train_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=2400)
        for row in train
    ]

    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
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
        train, train_vectors, scope="company", settings=settings, schema=schema, client=client
    )

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    baseline_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    nbsvm_oof = {c: np.full((len(train), len(classes)), -1e9, dtype=np.float64) for c in CS}
    for fit_idx, val_idx in folds.split(train_texts, labels):
        fold_texts = [train_texts[i] for i in fit_idx]
        fold_labels = [labels[i] for i in fit_idx]
        eval_texts = [train_texts[i] for i in val_idx]
        baseline = fit_with_descriptions(
            fold_texts, fold_labels, desc_texts, desc_labels, repeat=1
        )
        baseline_oof[val_idx] = decision_scores(baseline, eval_texts, classes)
        augmented_texts = fold_texts + desc_texts
        augmented_labels = fold_labels + desc_labels
        for c in CS:
            nbsvm_oof[c][val_idx] = nbsvm_scores(
                augmented_texts, augmented_labels, eval_texts, classes, c=c
            )

    baseline_pred = constrained(classes, baseline_oof, train_routes, train_recalls, schema)
    baseline_metric = metric(labels, baseline_pred)
    search: dict[str, dict[str, float]] = {}
    for c in CS:
        pred = constrained(classes, nbsvm_oof[c], train_routes, train_recalls, schema)
        search[f"C={c}"] = metric(labels, pred)
    selected_key = max(search, key=lambda key: search[key]["macro_f1"])
    selected_c = float(selected_key.split("=")[-1])
    selected_oof = search[selected_key]
    oof_gain = round(selected_oof["macro_f1"] - baseline_metric["macro_f1"], 6)

    report: dict[str, object] = {
        "scope": "company",
        "protocol": "production-like no-subject NB-SVM char reranker; C selected only by 3-fold train OOF; schema description x1 and exact-subject recall fallback k5 fixed from prior train-only recipe; external is touched only if OOF gain >= 0.005",
        "baseline_oof": baseline_metric,
        "nbsvm_oof_search": search,
        "selected_key": selected_key,
        "selected_oof": selected_oof,
        "oof_gain_vs_baseline": oof_gain,
        "external_gate_gain": OOF_GATE_GAIN,
        "external_touched": False,
        "production_like_external_reference_macro_f1": 0.770798,
    }
    if oof_gain >= OOF_GATE_GAIN:
        test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
        test_labels = [str(row.event_label) for row in test]
        test_texts = [
            routed_text(row, None, mode="no_subject", max_content_chars=2400)
            for row in test
        ]
        test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
        if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
            raise ValueError("test embedding order mismatch")
        test_routes, test_recalls = build_routes_and_recalls(
            test, test_vectors, scope="company", settings=settings, schema=schema, client=client
        )
        test_scores = nbsvm_scores(
            train_texts + desc_texts,
            labels + desc_labels,
            test_texts,
            classes,
            c=selected_c,
        )
        test_pred = constrained(classes, test_scores, test_routes, test_recalls, schema)
        external = metric(test_labels, test_pred)
        report.update(
            {
                "external_touched": True,
                "external": external,
                "external_gain_vs_reference": round(external["macro_f1"] - 0.770798, 6),
                "gate_0_80_passed": bool(external["macro_f1"] >= 0.80),
                "stretch_0_85_passed": bool(external["macro_f1"] >= 0.85),
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
