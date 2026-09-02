from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_pretrained_reranker_cascade import constrained
from benchmark_routed_subject_svc_v2 import metric, routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.confusion_specialist import (
    canonical_pair,
    confusion_pair_counts,
    select_confusion_pairs,
)
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel


MIN_COUNTS = (3, 5, 8)
MARGIN_QUANTILES = (0.2, 0.4, 0.6)


def _fold_count(labels: list[str]) -> int:
    minimum = min(Counter(labels).values())
    if minimum < 2:
        raise ValueError("specialist OOF requires at least two examples per class")
    return min(3, minimum)


def _oof_scores(
    texts: list[str],
    labels: list[str],
    classes: list[str],
    desc_texts: list[str],
    desc_labels: list[str],
    *,
    random_state: int,
) -> np.ndarray:
    output = np.full((len(texts), len(classes)), -1e9, dtype=np.float64)
    splitter = StratifiedKFold(
        n_splits=_fold_count(labels),
        shuffle=True,
        random_state=random_state,
    )
    text_array = np.asarray(texts, dtype=object)
    label_array = np.asarray(labels, dtype=object)
    for fit_idx, val_idx in splitter.split(text_array, label_array):
        model = fit_with_descriptions(
            text_array[fit_idx].tolist(),
            label_array[fit_idx].tolist(),
            desc_texts,
            desc_labels,
            repeat=1,
        )
        output[val_idx] = decision_scores(model, text_array[val_idx].tolist(), classes)
    return output


def _top2_pairs(scores: np.ndarray, classes: list[str]) -> list[tuple[str, str]]:
    order = np.argsort(-scores, axis=1)[:, :2]
    return [canonical_pair(classes[int(row[0])], classes[int(row[1])]) for row in order]


def _margins(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)[:, :2]
    rows = np.arange(len(scores))
    return scores[rows, order[:, 0]] - scores[rows, order[:, 1]]


def _train_specialists(
    texts: list[str],
    labels: list[str],
    selected_pairs: set[tuple[str, str]],
    desc_texts: list[str],
    desc_labels: list[str],
):
    description_map = dict(zip(desc_labels, desc_texts))
    models = {}
    for pair in selected_pairs:
        indices = [i for i, label in enumerate(labels) if label in pair]
        if len({labels[i] for i in indices}) != 2:
            continue
        pair_desc_texts = [description_map[label] for label in pair if label in description_map]
        pair_desc_labels = [label for label in pair if label in description_map]
        models[pair] = fit_with_descriptions(
            [texts[i] for i in indices],
            [labels[i] for i in indices],
            pair_desc_texts,
            pair_desc_labels,
            repeat=1,
        )
    return models


def _apply_specialists(
    scores: np.ndarray,
    texts: list[str],
    classes: list[str],
    specialists,
    *,
    margin_threshold: float,
) -> tuple[np.ndarray, int]:
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    pairs = _top2_pairs(adjusted, classes)
    margins = _margins(adjusted)
    triggered = 0
    class_to_index = {label: index for index, label in enumerate(classes)}
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, (pair, margin) in enumerate(zip(pairs, margins)):
        if margin <= margin_threshold and pair in specialists:
            grouped.setdefault(pair, []).append(index)
    for pair, indices in grouped.items():
        predictions = specialists[pair].predict([texts[i] for i in indices])
        for index, prediction in zip(indices, predictions):
            adjusted[index, class_to_index[str(prediction)]] = float(adjusted[index].max() + 1.0)
            triggered += 1
    return adjusted, triggered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]

    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding order mismatch")
    if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
        raise ValueError("test embedding order mismatch")

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
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope="company", settings=settings, schema=schema, client=client
    )

    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    predictions_by_key = {
        f"count={count}:q={quantile}": [""] * len(train)
        for count in MIN_COUNTS
        for quantile in MARGIN_QUANTILES
    }
    trigger_counts = {key: 0 for key in predictions_by_key}

    for fold, (fit_idx, val_idx) in enumerate(outer.split(train_texts, labels)):
        fit_texts = [train_texts[i] for i in fit_idx]
        fit_labels = [labels[i] for i in fit_idx]
        fit_routes = [train_routes[i] for i in fit_idx]
        fit_recalls = [train_recalls[i] for i in fit_idx]
        inner_scores = _oof_scores(
            fit_texts,
            fit_labels,
            classes,
            desc_texts,
            desc_labels,
            random_state=settings.model.random_state + fold + 1,
        )
        inner_pred = constrained(classes, inner_scores, fit_routes, fit_recalls, schema)
        pair_counts = confusion_pair_counts(fit_labels, inner_pred)
        inner_margins = _margins(inner_scores)

        global_model = fit_with_descriptions(
            fit_texts,
            fit_labels,
            desc_texts,
            desc_labels,
            repeat=1,
        )
        val_texts = [train_texts[i] for i in val_idx]
        val_scores = decision_scores(global_model, val_texts, classes)
        val_routes = [train_routes[i] for i in val_idx]
        val_recalls = [train_recalls[i] for i in val_idx]

        for count in MIN_COUNTS:
            selected_pairs = select_confusion_pairs(pair_counts, minimum_count=count)
            specialists = _train_specialists(
                fit_texts,
                fit_labels,
                selected_pairs,
                desc_texts,
                desc_labels,
            )
            for quantile in MARGIN_QUANTILES:
                threshold = float(np.quantile(inner_margins, quantile))
                adjusted, triggered = _apply_specialists(
                    val_scores,
                    val_texts,
                    classes,
                    specialists,
                    margin_threshold=threshold,
                )
                pred = constrained(classes, adjusted, val_routes, val_recalls, schema)
                key = f"count={count}:q={quantile}"
                trigger_counts[key] += triggered
                for index, value in zip(val_idx, pred):
                    predictions_by_key[key][int(index)] = value

    oof_results = {
        key: {**metric(labels, pred), "triggered": trigger_counts[key]}
        for key, pred in predictions_by_key.items()
    }
    selected_key = max(oof_results, key=lambda key: oof_results[key]["macro_f1"])
    count_value = int(selected_key.split(":")[0].split("=")[1])
    quantile_value = float(selected_key.split(":")[1].split("=")[1])

    full_oof = _oof_scores(
        train_texts,
        labels,
        classes,
        desc_texts,
        desc_labels,
        random_state=settings.model.random_state,
    )
    full_oof_pred = constrained(classes, full_oof, train_routes, train_recalls, schema)
    pair_counts = confusion_pair_counts(labels, full_oof_pred)
    selected_pairs = select_confusion_pairs(pair_counts, minimum_count=count_value)
    specialists = _train_specialists(
        train_texts,
        labels,
        selected_pairs,
        desc_texts,
        desc_labels,
    )
    margin_threshold = float(np.quantile(_margins(full_oof), quantile_value))

    full_model = fit_with_descriptions(train_texts, labels, desc_texts, desc_labels, repeat=1)
    test_scores = decision_scores(full_model, test_texts, classes)
    baseline_external = constrained(classes, test_scores, test_routes, test_recalls, schema)
    adjusted_test, external_triggered = _apply_specialists(
        test_scores,
        test_texts,
        classes,
        specialists,
        margin_threshold=margin_threshold,
    )
    external_pred = constrained(classes, adjusted_test, test_routes, test_recalls, schema)

    payload = {
        "scope": "company",
        "protocol": "production-like nested-OOF automatic confusion-pair char-SVC specialists; pair count and margin quantile selected only by train OOF; external used once after selection",
        "oof_baseline": metric(labels, full_oof_pred),
        "oof_search": dict(sorted(oof_results.items(), key=lambda item: item[1]["macro_f1"], reverse=True)),
        "selected_key": selected_key,
        "selected_pair_count": len(selected_pairs),
        "selected_margin_threshold": round(margin_threshold, 6),
        "external_baseline": metric(test_labels, baseline_external),
        "external": metric(test_labels, external_pred),
        "external_triggered": external_triggered,
    }
    payload["gain_vs_external_baseline"] = round(
        payload["external"]["macro_f1"] - payload["external_baseline"]["macro_f1"], 6
    )
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
