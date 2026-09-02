from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from benchmark_routed_subject_svc_v2 import predict_with_policy
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.preprocess import clean_text


SUBJECT_TOKEN = "主体占位"


def public_subject_aliases(schema: EventSchemaIndex, *, scope: str) -> list[str]:
    aliases = {
        clean_text(definition.subject_name)
        for definition in schema.definitions
        if definition.scope == scope and len(clean_text(definition.subject_name)) >= 2
    }
    return sorted(aliases, key=lambda value: (-len(value), value))


def mask_public_subject_names(text: str, aliases: list[str]) -> str:
    masked = clean_text(text)
    for alias in aliases:
        masked = masked.replace(alias, SUBJECT_TOKEN)
    return re.sub(rf"(?:{SUBJECT_TOKEN}\s*)+", SUBJECT_TOKEN, masked).strip()


def metric(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def production_text(article, *, max_content_chars: int, aliases: list[str] | None) -> str:
    title = clean_text(article.title)
    source = clean_text(article.source)
    content = clean_text(article.content[:max_content_chars])
    if aliases is not None:
        title = mask_public_subject_names(title, aliases)
        content = mask_public_subject_names(content, aliases)
    return " ".join(part for part in [title, source, content] if part)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-oof-gain-for-external", type=float, default=0.005)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
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
    aliases = public_subject_aliases(schema, scope="company")
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
    description_texts, description_labels = label_description_examples(schema, scope="company")
    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)

    text_modes = ["baseline", "public_subject_name_mask", "accepted_subject_name_mask"]
    oof = {}
    oof_predictions = {}
    for mode in text_modes:
        texts = []
        for row, route in zip(train, train_routes):
            mode_aliases = None
            if mode == "public_subject_name_mask":
                mode_aliases = aliases
            elif mode == "accepted_subject_name_mask" and route.accepted_subject_name:
                mode_aliases = [route.accepted_subject_name]
            texts.append(production_text(row, max_content_chars=2400, aliases=mode_aliases))
        scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
        for fit_idx, val_idx in folds.split(texts, labels):
            model = fit_with_descriptions(
                [texts[i] for i in fit_idx],
                [labels[i] for i in fit_idx],
                description_texts,
                description_labels,
                repeat=1,
            )
            scores[val_idx] = decision_scores(model, [texts[i] for i in val_idx], classes)
        pred = predict_with_policy(
            classes, scores, train_routes, train_recalls, schema, scope="company", policy_name="exact_fallback_k5"
        )
        oof[mode] = metric(labels, pred)
        oof_predictions[mode] = pred

    selected_mode = max(
        ["public_subject_name_mask", "accepted_subject_name_mask"],
        key=lambda mode: oof[mode]["macro_f1"],
    )
    oof_gain = round(oof[selected_mode]["macro_f1"] - oof["baseline"]["macro_f1"], 6)
    external = None
    if oof_gain >= args.min_oof_gain_for_external:
        train_texts = []
        test_texts = []
        for row, route in zip(train, train_routes):
            mode_aliases = aliases if selected_mode == "public_subject_name_mask" else ([route.accepted_subject_name] if route.accepted_subject_name else None)
            train_texts.append(production_text(row, max_content_chars=2400, aliases=mode_aliases))
        for row, route in zip(test, test_routes):
            mode_aliases = aliases if selected_mode == "public_subject_name_mask" else ([route.accepted_subject_name] if route.accepted_subject_name else None)
            test_texts.append(production_text(row, max_content_chars=2400, aliases=mode_aliases))
        model = fit_with_descriptions(
            train_texts, labels, description_texts, description_labels, repeat=1
        )
        test_scores = decision_scores(model, test_texts, classes)
        pred = predict_with_policy(
            classes, test_scores, test_routes, test_recalls, schema, scope="company", policy_name="exact_fallback_k5"
        )
        external = metric(test_labels, pred)

    payload = {
        "protocol": "production-like company subject-shortcut ablation; only public schema subject names may be masked; 3-fold train OOF decides whether external is touched",
        "alias_count": len(aliases),
        "fixed_recipe": "2400 chars + schema desc x1 + exact_fallback_k5",
        "oof": oof,
        "selected_mask_mode": selected_mode,
        "oof_gain": oof_gain,
        "min_oof_gain_for_external": args.min_oof_gain_for_external,
        "external_evaluated": external is not None,
        "external": external,
        "production_like_reference_external_macro_f1": 0.770798,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
