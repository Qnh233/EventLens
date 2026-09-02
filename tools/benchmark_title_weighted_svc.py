from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_pretrained_reranker_cascade import constrained
from benchmark_routed_subject_svc_v2 import metric
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


TITLE_REPEATS = (1, 2, 3)
CONTENT_LIMITS = (1200, 1800, 2400)


def article_text(article, *, title_repeat: int, content_limit: int) -> str:
    """只重加权可生产获得的标题/正文，不引入主体真值字段。"""
    title = clean_text(article.title)
    content = clean_text(article.content)[:content_limit]
    return "。".join([title] * title_repeat + [content])


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

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    search = {}
    for title_repeat in TITLE_REPEATS:
        for content_limit in CONTENT_LIMITS:
            texts = [article_text(row, title_repeat=title_repeat, content_limit=content_limit) for row in train]
            oof_scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
            for fit_idx, val_idx in splitter.split(texts, labels):
                fit_texts = [texts[i] for i in fit_idx]
                fit_labels = [labels[i] for i in fit_idx]
                model = fit_with_descriptions(fit_texts, fit_labels, desc_texts, desc_labels, repeat=1)
                oof_scores[val_idx] = decision_scores(model, [texts[i] for i in val_idx], classes)
            pred = constrained(classes, oof_scores, train_routes, train_recalls, schema)
            search[f"title_repeat={title_repeat}:content={content_limit}"] = metric(labels, pred)

    selected_key = max(search, key=lambda key: search[key]["macro_f1"])
    parts = dict(part.split("=") for part in selected_key.split(":"))
    title_repeat = int(parts["title_repeat"])
    content_limit = int(parts["content"])

    train_texts = [article_text(row, title_repeat=title_repeat, content_limit=content_limit) for row in train]
    test_texts = [article_text(row, title_repeat=title_repeat, content_limit=content_limit) for row in test]
    model = fit_with_descriptions(train_texts, labels, desc_texts, desc_labels, repeat=1)
    test_scores = decision_scores(model, test_texts, classes)
    external_pred = constrained(classes, test_scores, test_routes, test_recalls, schema)
    external = metric(test_labels, external_pred)

    report = {
        "scope": "company",
        "protocol": "production-like title/body reweighting char-SVC; title repeat and content limit selected only by 3-fold train OOF; external used once after selection",
        "oof_search": search,
        "selected_key": selected_key,
        "external": external,
        "production_like_svc_reference_macro_f1": 0.770798,
        "gain_vs_reference": round(external["macro_f1"] - 0.770798, 6),
        "gate_macro_f1": 0.8,
        "stretch_target_macro_f1": 0.85,
        "gate_passed": bool(external["macro_f1"] >= 0.8),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
