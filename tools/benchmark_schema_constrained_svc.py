from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import (
    EventSchemaIndex,
    NativeSentenceTransformerEmbeddingClient,
    SubjectConstrainedEventRetriever,
)
from eventlens.io import read_competition_labeled_excel
from eventlens.preprocess import build_model_text
from eventlens.schema_constrained_classifier import constrain_predictions
from eventlens.subject_routing import SubjectRouter, SubjectRoutingPolicy


POLICY_SPECS = [
    ("global", "global", 3),
    ("hard_subject", "hard_subject", 3),
    ("hard_subject_recall_k3", "hard_subject_recall", 3),
    ("hard_subject_recall_fallback_k3", "hard_subject_recall_fallback", 3),
    ("hard_subject_recall_fallback_k5", "hard_subject_recall_fallback", 5),
    ("exact_subject_recall_fallback_k3", "exact_subject_recall_fallback", 3),
    ("exact_subject_recall_fallback_k5", "exact_subject_recall_fallback", 5),
]
AUGMENT_REPEATS = [0, 1, 2, 4]


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                    max_features=80000,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42)),
        ]
    )


def metric(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def decision_scores(model: Pipeline, texts: list[str], global_classes: list[str]) -> np.ndarray:
    local_scores = np.asarray(model.decision_function(texts), dtype=np.float64)
    local_classes = [str(x) for x in model.named_steps["clf"].classes_]
    output = np.full((len(texts), len(global_classes)), -1e9, dtype=np.float64)
    class_to_index = {label: index for index, label in enumerate(global_classes)}
    for local_index, label in enumerate(local_classes):
        output[:, class_to_index[label]] = local_scores[:, local_index]
    return output


def label_description_examples(schema: EventSchemaIndex, *, scope: str) -> tuple[list[str], list[str]]:
    grouped: dict[str, list[str]] = {}
    for definition in schema.definitions:
        if definition.scope != scope:
            continue
        descriptions = grouped.setdefault(definition.event_name, [])
        if definition.description and definition.description not in descriptions:
            descriptions.append(definition.description)
    texts: list[str] = []
    labels: list[str] = []
    for label in sorted(grouped):
        description = "；".join(grouped[label][:5])
        texts.append(f"事件类型：{label}。事件定义：{description}")
        labels.append(label)
    return texts, labels


def fit_with_descriptions(
    train_texts: list[str],
    train_labels: list[str],
    description_texts: list[str],
    description_labels: list[str],
    *,
    repeat: int,
) -> Pipeline:
    model = build_model()
    model.fit(
        train_texts + description_texts * repeat,
        train_labels + description_labels * repeat,
    )
    return model


def build_routes_and_recalls(articles, vectors, *, scope, settings, schema, client):
    policy_cfg = getattr(settings.subject_routing, scope)
    router = SubjectRouter(
        schema,
        client,
        max_query_chars=settings.subject_routing.max_query_chars,
        min_alias_chars=settings.subject_routing.min_alias_chars,
    )
    routes = router.route_from_vectors(
        articles,
        vectors,
        scope=scope,
        policy=SubjectRoutingPolicy(**policy_cfg.model_dump()),
    )
    subject_codes = [
        [row.accepted_subject_code]
        if row.accepted_subject_code
        else [candidate.subject_code for candidate in row.candidates]
        for row in routes
    ]
    retriever = SubjectConstrainedEventRetriever(
        schema,
        client,
        max_query_chars=settings.event_retrieval.max_query_chars,
    )
    recalls = retriever.recall_from_vectors(
        [row.article_id for row in articles],
        vectors,
        subject_codes,
        scope=scope,
        top_k=5,
    )
    return routes, recalls


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
    description_texts, description_labels = label_description_examples(
        schema, scope=args.scope
    )
    train_routes, train_recalls = build_routes_and_recalls(
        train, train_vectors, scope=args.scope, settings=settings, schema=schema, client=client
    )
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope=args.scope, settings=settings, schema=schema, client=client
    )

    train_labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    global_classes = sorted(set(train_labels))
    max_chars = settings.model.text.max_content_chars
    train_texts = [build_model_text(row, max_chars) for row in train]
    test_texts = [build_model_text(row, max_chars) for row in test]

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    oof_results: dict[str, dict] = {}
    winner_key = ""
    winner_score = -1.0
    winner_spec = POLICY_SPECS[0]
    winner_repeat = 0
    for repeat in AUGMENT_REPEATS:
        oof_scores = np.full((len(train), len(global_classes)), -1e9, dtype=np.float64)
        for train_idx, val_idx in folds.split(train_texts, train_labels):
            model = fit_with_descriptions(
                [train_texts[i] for i in train_idx],
                [train_labels[i] for i in train_idx],
                description_texts,
                description_labels,
                repeat=repeat,
            )
            oof_scores[val_idx] = decision_scores(
                model, [train_texts[i] for i in val_idx], global_classes
            )
        for name, policy, recall_top_k in POLICY_SPECS:
            pred = constrain_predictions(
                global_classes,
                oof_scores,
                train_routes,
                train_recalls,
                schema,
                scope=args.scope,
                policy=policy,
                recall_top_k=recall_top_k,
            )
            key = f"desc_x{repeat}:{name}"
            score = metric(train_labels, pred)
            oof_results[key] = score
            if score["macro_f1"] > winner_score:
                winner_key = key
                winner_score = score["macro_f1"]
                winner_spec = (name, policy, recall_top_k)
                winner_repeat = repeat

    final_model = fit_with_descriptions(
        train_texts,
        train_labels,
        description_texts,
        description_labels,
        repeat=winner_repeat,
    )
    test_scores = decision_scores(final_model, test_texts, global_classes)
    external_results = {}
    for name, policy, recall_top_k in POLICY_SPECS:
        pred = constrain_predictions(
            global_classes,
            test_scores,
            test_routes,
            test_recalls,
            schema,
            scope=args.scope,
            policy=policy,
            recall_top_k=recall_top_k,
        )
        external_results[name] = metric(test_labels, pred)

    selected_external = external_results[winner_spec[0]]
    payload = {
        "scope": args.scope,
        "selection_protocol": "3-fold OOF on tagged train selects label-description repeat and constraint policy; tagged test is external validation",
        "label_description_count": len(description_texts),
        "oof": oof_results,
        "selected_key": winner_key,
        "selected_description_repeat": winner_repeat,
        "selected_policy": winner_spec[0],
        "selected_policy_spec": {
            "policy": winner_spec[1],
            "recall_top_k": winner_spec[2],
        },
        "external": external_results,
        "selected_external": selected_external,
        "gate_macro_f1": 0.80,
        "gate_passed": selected_external["macro_f1"] >= 0.80,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
