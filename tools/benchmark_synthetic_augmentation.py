from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
from eventlens.preprocess import clean_text


BUDGETS = (0, 1, 2, 4, 8)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def synthetic_text(row: dict, *, max_content_chars: int = 2400) -> str:
    return " ".join(
        clean_text(value)
        for value in (
            row.get("title", ""),
            row.get("source", "synthetic_schema_deepseek"),
            str(row.get("content", ""))[:max_content_chars],
        )
        if clean_text(value)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    parser.add_argument("--synthetic", required=True)
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

    synthetic_rows = [
        json.loads(line)
        for line in Path(args.synthetic).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    desc_texts, desc_labels = label_description_examples(schema, scope=args.scope)
    desc_vectors = np.asarray(client.embed(desc_texts), dtype=np.float32)
    synth_texts_all = [synthetic_text(row) for row in synthetic_rows]
    synth_vectors = np.asarray(client.embed(synth_texts_all), dtype=np.float32) if synth_texts_all else np.empty((0, desc_vectors.shape[1]), dtype=np.float32)
    similarities = _normalize(synth_vectors) @ _normalize(desc_vectors).T if len(synth_vectors) else np.empty((0, len(desc_labels)))
    accepted_by_label: dict[str, list[dict]] = defaultdict(list)
    for i, row in enumerate(synthetic_rows):
        if similarities.shape[0] == 0:
            break
        order = np.argsort(-similarities[i])
        top1 = desc_labels[int(order[0])]
        margin = float(similarities[i, int(order[0])] - similarities[i, int(order[1])]) if len(order) > 1 else 1.0
        if top1 != str(row.get("event_label", "")):
            continue
        enriched = dict(row)
        enriched["bge_label_score"] = round(float(similarities[i, int(order[0])]), 6)
        enriched["bge_label_margin"] = round(margin, 6)
        accepted_by_label[top1].append(enriched)
    for rows in accepted_by_label.values():
        rows.sort(key=lambda row: (-row["bge_label_margin"], -row["bge_label_score"], row["title"]))

    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    oof = {}
    selected_budget = 0
    selected_score = -1.0
    for budget in BUDGETS:
        extra_rows = [row for label in sorted(accepted_by_label) for row in accepted_by_label[label][:budget]]
        extra_texts = [synthetic_text(row) for row in extra_rows]
        extra_labels = [str(row["event_label"]) for row in extra_rows]
        scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
        for train_idx, val_idx in folds.split(train_texts, labels):
            model = fit_with_descriptions(
                [train_texts[i] for i in train_idx] + extra_texts,
                [labels[i] for i in train_idx] + extra_labels,
                desc_texts,
                desc_labels,
                repeat=1,
            )
            scores[val_idx] = decision_scores(model, [train_texts[i] for i in val_idx], classes)
        pred = predict_with_policy(
            classes,
            scores,
            train_routes,
            train_recalls,
            schema,
            scope=args.scope,
            policy_name="exact_fallback_k5",
        )
        oof[str(budget)] = metric(labels, pred)
        if oof[str(budget)]["macro_f1"] > selected_score:
            selected_budget = budget
            selected_score = oof[str(budget)]["macro_f1"]

    selected_rows = [
        row for label in sorted(accepted_by_label) for row in accepted_by_label[label][:selected_budget]
    ]
    selected_texts = [synthetic_text(row) for row in selected_rows]
    selected_labels = [str(row["event_label"]) for row in selected_rows]
    model = fit_with_descriptions(
        train_texts + selected_texts,
        labels + selected_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    test_scores = decision_scores(model, test_texts, classes)
    test_pred = predict_with_policy(
        classes,
        test_scores,
        test_routes,
        test_recalls,
        schema,
        scope=args.scope,
        policy_name="exact_fallback_k5",
    )
    external = metric(test_labels, test_pred)
    payload = {
        "scope": args.scope,
        "protocol": "schema-only DeepSeek synthetic augmentation; BGE label-description top1 filter; train OOF selects per-label budget; production-like external test",
        "generated_count": len(synthetic_rows),
        "accepted_count": sum(len(rows) for rows in accepted_by_label.values()),
        "accepted_per_label": {label: len(rows) for label, rows in sorted(accepted_by_label.items())},
        "selected_budget_per_label": selected_budget,
        "selected_synthetic_count": len(selected_rows),
        "selected_oof": oof[str(selected_budget)],
        "external": external,
        "gate_macro_f1": 0.80,
        "gate_passed": external["macro_f1"] >= 0.80,
        "oof": oof,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
