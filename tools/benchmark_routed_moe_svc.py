from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_routed_subject_svc_v2 import metric, predict_with_policy, routed_text
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
from eventlens.preprocess import build_model_text, clean_text


GATES = ("global_only", "exact_alias", "hard_route")
POST_POLICIES = ("global", "exact_fallback_k5")


def routed_subject_text(article, route, *, max_content_chars: int) -> str:
    subject = route.accepted_subject_name or ""
    parts = [article.title, subject, article.source, article.content[:max_content_chars]]
    return " ".join(clean_text(part) for part in parts if clean_text(part))


def choose_scores(
    global_scores: np.ndarray,
    subject_scores: np.ndarray,
    routes,
    *,
    gate: str,
) -> np.ndarray:
    output = np.asarray(global_scores, dtype=np.float64).copy()
    if gate == "global_only":
        return output
    for i, route in enumerate(routes):
        use_subject = (
            route.method == "exact_alias"
            if gate == "exact_alias"
            else route.accepted_subject_code is not None
        )
        if use_subject:
            output[i] = subject_scores[i]
    return output


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

    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    desc_texts, desc_labels = label_description_examples(schema, scope=args.scope)
    global_train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    global_test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]
    subject_train_texts = [build_model_text(row, max_content_chars=1200) for row in train]
    subject_validation_texts = [
        routed_subject_text(row, route, max_content_chars=1200)
        for row, route in zip(train, train_routes)
    ]
    subject_test_texts = [
        routed_subject_text(row, route, max_content_chars=1200)
        for row, route in zip(test, test_routes)
    ]

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    global_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    subject_oof = np.full_like(global_oof, -1e9)
    for train_idx, val_idx in folds.split(global_train_texts, labels):
        global_model = fit_with_descriptions(
            [global_train_texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        subject_model = fit_with_descriptions(
            [subject_train_texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            desc_texts,
            desc_labels,
            repeat=0,
        )
        global_oof[val_idx] = decision_scores(global_model, [global_train_texts[i] for i in val_idx], classes)
        subject_oof[val_idx] = decision_scores(subject_model, [subject_validation_texts[i] for i in val_idx], classes)

    oof = {}
    selected_key = ""
    selected_score = -1.0
    selected_gate = ""
    selected_policy = ""
    for gate in GATES:
        mixed = choose_scores(global_oof, subject_oof, train_routes, gate=gate)
        for post_policy in POST_POLICIES:
            pred = predict_with_policy(
                classes,
                mixed,
                train_routes,
                train_recalls,
                schema,
                scope=args.scope,
                policy_name=post_policy,
            )
            key = f"{gate}:{post_policy}"
            oof[key] = metric(labels, pred)
            if oof[key]["macro_f1"] > selected_score:
                selected_key = key
                selected_score = oof[key]["macro_f1"]
                selected_gate = gate
                selected_policy = post_policy

    global_model = fit_with_descriptions(
        global_train_texts, labels, desc_texts, desc_labels, repeat=1
    )
    subject_model = fit_with_descriptions(
        subject_train_texts, labels, desc_texts, desc_labels, repeat=0
    )
    global_test_scores = decision_scores(global_model, global_test_texts, classes)
    subject_test_scores = decision_scores(subject_model, subject_test_texts, classes)
    mixed_test_scores = choose_scores(
        global_test_scores, subject_test_scores, test_routes, gate=selected_gate
    )
    test_pred = predict_with_policy(
        classes,
        mixed_test_scores,
        test_routes,
        test_recalls,
        schema,
        scope=args.scope,
        policy_name=selected_policy,
    )
    external = metric(test_labels, test_pred)
    payload = {
        "scope": args.scope,
        "protocol": "production-like routed mixture-of-experts; true subject used only for training subject expert; inference uses BGE/alias route; train OOF selects gate",
        "selected_key": selected_key,
        "selected_gate": selected_gate,
        "selected_post_policy": selected_policy,
        "selected_oof": oof[selected_key],
        "external": external,
        "gate_macro_f1": 0.80,
        "gate_passed": external["macro_f1"] >= 0.80,
        "oof": dict(sorted(oof.items(), key=lambda item: item[1]["macro_f1"], reverse=True)),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
