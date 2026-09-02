from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

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
from eventlens.preprocess import clean_text
from eventlens.schema_constrained_classifier import constrain_predictions


@dataclass(frozen=True)
class TextConfig:
    mode: str
    max_content_chars: int
    description_repeat: int

    @property
    def key(self) -> str:
        return f"{self.mode}:chars{self.max_content_chars}:desc{self.description_repeat}"


POLICIES = {
    "global": None,
    "exact_fallback_k5": ("exact_subject_recall_fallback", 5),
    "hard_recall_k3": ("hard_subject_recall", 3),
}


def routed_text(article, route, *, mode: str, max_content_chars: int) -> str:
    if mode == "no_subject":
        subject = ""
    elif mode == "route_hard":
        subject = route.accepted_subject_name or ""
    elif mode == "route_top3":
        names = (
            [route.accepted_subject_name]
            if route.accepted_subject_name
            else [row.subject_name for row in route.candidates[:3]]
        )
        subject = " ".join(name for name in names if name)
    else:
        raise ValueError(f"未知 routed text mode: {mode}")
    parts = [article.title, subject, article.source, article.content[:max_content_chars]]
    return " ".join(clean_text(part) for part in parts if clean_text(part))


def metric(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def predict_with_policy(classes, scores, routes, recalls, schema, *, scope, policy_name):
    spec = POLICIES[policy_name]
    if spec is None:
        return [classes[index] for index in np.argmax(scores, axis=1)]
    policy, recall_top_k = spec
    return constrain_predictions(
        classes,
        scores,
        routes,
        recalls,
        schema,
        scope=scope,
        policy=policy,
        recall_top_k=recall_top_k,
    )


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
    train_routes, train_recalls = build_routes_and_recalls(
        train, train_vectors, scope=args.scope, settings=settings, schema=schema, client=client
    )
    test_routes, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope=args.scope, settings=settings, schema=schema, client=client
    )

    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    description_texts, description_labels = label_description_examples(schema, scope=args.scope)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    configs = [
        TextConfig(mode, chars, repeat)
        for mode in ["no_subject", "route_hard", "route_top3"]
        for chars in [1200, 2400]
        for repeat in [0, 1]
    ]

    oof: dict[str, dict[str, float]] = {}
    winner_key = ""
    winner_score = -1.0
    winner_config: TextConfig | None = None
    winner_policy = ""
    for config in configs:
        texts = [
            routed_text(row, route, mode=config.mode, max_content_chars=config.max_content_chars)
            for row, route in zip(train, train_routes)
        ]
        scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
        for train_idx, val_idx in folds.split(texts, labels):
            model = fit_with_descriptions(
                [texts[i] for i in train_idx],
                [labels[i] for i in train_idx],
                description_texts,
                description_labels,
                repeat=config.description_repeat,
            )
            scores[val_idx] = decision_scores(model, [texts[i] for i in val_idx], classes)
        for policy_name in POLICIES:
            pred = predict_with_policy(
                classes,
                scores,
                train_routes,
                train_recalls,
                schema,
                scope=args.scope,
                policy_name=policy_name,
            )
            key = f"{config.key}:{policy_name}"
            oof[key] = metric(labels, pred)
            if oof[key]["macro_f1"] > winner_score:
                winner_score = oof[key]["macro_f1"]
                winner_key = key
                winner_config = config
                winner_policy = policy_name

    assert winner_config is not None
    train_texts = [
        routed_text(row, route, mode=winner_config.mode, max_content_chars=winner_config.max_content_chars)
        for row, route in zip(train, train_routes)
    ]
    test_texts = [
        routed_text(row, route, mode=winner_config.mode, max_content_chars=winner_config.max_content_chars)
        for row, route in zip(test, test_routes)
    ]
    model = fit_with_descriptions(
        train_texts,
        labels,
        description_texts,
        description_labels,
        repeat=winner_config.description_repeat,
    )
    test_scores = decision_scores(model, test_texts, classes)
    test_pred = predict_with_policy(
        classes,
        test_scores,
        test_routes,
        test_recalls,
        schema,
        scope=args.scope,
        policy_name=winner_policy,
    )
    external = metric(test_labels, test_pred)
    payload = {
        "scope": args.scope,
        "protocol": "production-like: true subject fields removed; BGE route/schema only; all choices selected by train OOF",
        "selected_key": winner_key,
        "selected_config": winner_config.__dict__,
        "selected_policy": winner_policy,
        "selected_oof": oof[winner_key],
        "external": external,
        "gate_macro_f1": 0.80,
        "gate_passed": external["macro_f1"] >= 0.80,
        "oof": dict(sorted(oof.items(), key=lambda item: item[1]["macro_f1"], reverse=True)),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
