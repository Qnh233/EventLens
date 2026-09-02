from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from benchmark_routed_subject_svc_v2 import metric, predict_with_policy, routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel


def class_weights(labels: list[str], *, power: float) -> dict[str, float] | None:
    if power == 0.0:
        return None
    counts = Counter(labels)
    n = len(labels)
    k = len(counts)
    raw = {label: (n / (k * count)) ** power for label, count in counts.items()}
    sample_mean = sum(counts[label] * weight for label, weight in raw.items()) / n
    return {label: weight / sample_mean for label, weight in raw.items()}


def aligned_scores(model: LinearSVC, matrix, classes: list[str]) -> np.ndarray:
    local = np.asarray(model.decision_function(matrix), dtype=np.float64)
    output = np.full((matrix.shape[0], len(classes)), -1e9, dtype=np.float64)
    class_to_index = {label: index for index, label in enumerate(classes)}
    for local_index, label in enumerate(model.classes_):
        output[:, class_to_index[str(label)]] = local[:, local_index]
    return output


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=80000,
        sublinear_tf=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company"], default="company")
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [
        row.article_id for row in train
    ]:
        raise ValueError("train embedding 顺序不一致")
    if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [
        row.article_id for row in test
    ]:
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
        train, train_vectors, scope="company", settings=settings, schema=schema, client=client
    )
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope="company", settings=settings, schema=schema, client=client
    )

    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    description_texts, description_labels = label_description_examples(schema, scope="company")
    train_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train
    ]
    test_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test
    ]

    powers = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    c_values = (0.25, 0.5, 1.0, 2.0, 4.0)
    params = [(power, c_value) for power in powers for c_value in c_values]
    oof_scores = {
        (power, c_value): np.full((len(train), len(classes)), -1e9, dtype=np.float64)
        for power, c_value in params
    }
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    for train_idx, val_idx in folds.split(train_texts, labels):
        fold_texts = [train_texts[i] for i in train_idx] + description_texts
        fold_labels = [labels[i] for i in train_idx] + description_labels
        vectorizer = build_vectorizer()
        x_train = vectorizer.fit_transform(fold_texts)
        x_val = vectorizer.transform([train_texts[i] for i in val_idx])
        for power, c_value in params:
            model = LinearSVC(
                C=c_value,
                class_weight=class_weights(fold_labels, power=power),
                random_state=settings.model.random_state,
            )
            model.fit(x_train, fold_labels)
            oof_scores[(power, c_value)][val_idx] = aligned_scores(model, x_val, classes)

    results: list[dict] = []
    winner: tuple[float, float] | None = None
    winner_score = -1.0
    for power, c_value in params:
        pred = predict_with_policy(
            classes,
            oof_scores[(power, c_value)],
            train_routes,
            train_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        scores = metric(labels, pred)
        results.append({"class_weight_power": power, "C": c_value, **scores})
        if scores["macro_f1"] > winner_score:
            winner = (power, c_value)
            winner_score = scores["macro_f1"]

    assert winner is not None
    winner_power, winner_c = winner
    final_texts = train_texts + description_texts
    final_labels = labels + description_labels
    vectorizer = build_vectorizer()
    x_train = vectorizer.fit_transform(final_texts)
    x_test = vectorizer.transform(test_texts)
    model = LinearSVC(
        C=winner_c,
        class_weight=class_weights(final_labels, power=winner_power),
        random_state=settings.model.random_state,
    )
    model.fit(x_train, final_labels)
    test_scores = aligned_scores(model, x_test, classes)
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
    payload = {
        "scope": "company",
        "protocol": "production-like no-subject SVC; train OOF selects class-weight power and C; exact-alias schema fallback fixed before external validation",
        "selected": {
            "class_weight_power": winner_power,
            "C": winner_c,
            "oof_macro_f1": winner_score,
        },
        "oof": sorted(results, key=lambda row: row["macro_f1"], reverse=True),
        "external": external,
        "gate_macro_f1": 0.80,
        "gate_passed": external["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
