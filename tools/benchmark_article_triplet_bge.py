from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_routed_subject_svc_v2 import predict_with_policy, routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.article_contrastive import (
    ArticleTripletTrainingConfig,
    encode_articles,
    train_article_triplet_encoder,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.exemplar_reranker import topk_exemplar_scores
from eventlens.io import read_competition_labeled_excel


FUSION_WEIGHTS = (0.05, 0.1, 0.2)
TOP_K_VALUES = (1, 3)
PILOT_GAIN_GATE = 0.005


def metric(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def row_zscore(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / np.maximum(std, 1e-6)


def duplication_groups(articles) -> list[str]:
    return [
        str(row.duplication_id).strip()
        if row.duplication_id not in (None, "")
        else f"article:{row.article_id}"
        for row in articles
    ]


def confusion_negative_map(
    labels: list[str],
    predictions: list[str],
    classes: list[str],
    *,
    top_k: int = 3,
) -> dict[int, list[int]]:
    matrix = confusion_matrix(labels, predictions, labels=classes).astype(np.float64)
    row_total = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
    rates = matrix / row_total
    np.fill_diagonal(rates, -1.0)
    output = {}
    for class_id in range(len(classes)):
        order = np.argsort(-rates[class_id])
        output[class_id] = [int(index) for index in order if int(index) != class_id][:top_k]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in train]
    classes = sorted(set(labels))
    groups = duplication_groups(train)
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
    routes, recalls = build_routes_and_recalls(
        train,
        train_vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=client,
    )
    texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]

    # 先用一个固定 group-safe 80/20 pilot；只有明显正增益才值得进入昂贵 3-fold 训练。
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=settings.model.random_state)
    fit_idx, val_idx = next(outer.split(texts, labels, groups))
    fit_labels = [labels[i] for i in fit_idx]
    val_labels = [labels[i] for i in val_idx]

    svc = fit_with_descriptions(
        [texts[i] for i in fit_idx],
        fit_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    val_svc_scores = decision_scores(svc, [texts[i] for i in val_idx], classes)
    baseline_pred = predict_with_policy(
        classes,
        val_svc_scores,
        [routes[i] for i in val_idx],
        [recalls[i] for i in val_idx],
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    baseline = metric(val_labels, baseline_pred)

    # hard negative 只从 fit 子集内部 3-fold OOF 混淆得到。
    fit_groups = [groups[i] for i in fit_idx]
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    fit_oof = np.empty(len(fit_idx), dtype=object)
    fit_text_array = np.asarray([texts[i] for i in fit_idx], dtype=object)
    fit_label_array = np.asarray(fit_labels, dtype=object)
    fit_group_array = np.asarray(fit_groups, dtype=object)
    for inner_fit, inner_val in inner.split(fit_text_array, fit_label_array, fit_group_array):
        inner_svc = fit_with_descriptions(
            fit_text_array[inner_fit].tolist(),
            fit_label_array[inner_fit].tolist(),
            desc_texts,
            desc_labels,
            repeat=1,
        )
        fit_oof[inner_val] = inner_svc.predict(fit_text_array[inner_val].tolist())
    hard_map = confusion_negative_map(fit_labels, fit_oof.tolist(), classes, top_k=3)

    cfg = ArticleTripletTrainingConfig(
        model=args.model,
        max_length=384,
        max_content_chars=2400,
        batch_size=4,
        epochs=3,
        learning_rate=1e-5,
        margin=0.08,
        trainable_last_layers=2,
        random_state=settings.model.random_state,
    )
    trained = train_article_triplet_encoder(
        [train[i] for i in fit_idx],
        fit_labels,
        classes=classes,
        hard_negative_ids_by_class=hard_map,
        config=cfg,
        local_files_only=True,
    )
    fit_vectors = encode_articles(
        trained["model"],
        trained["tokenizer"],
        [train[i] for i in fit_idx],
        max_length=cfg.max_length,
        max_content_chars=cfg.max_content_chars,
        batch_size=16,
    )
    val_vectors = encode_articles(
        trained["model"],
        trained["tokenizer"],
        [train[i] for i in val_idx],
        max_length=cfg.max_length,
        max_content_chars=cfg.max_content_chars,
        batch_size=16,
    )

    search: dict[str, dict[str, float]] = {}
    best_key = "baseline"
    best_score = baseline["macro_f1"]
    for top_k in TOP_K_VALUES:
        semantic = topk_exemplar_scores(fit_vectors, fit_labels, val_vectors, classes, top_k=top_k)
        for weight in FUSION_WEIGHTS:
            fused = row_zscore(val_svc_scores) + weight * row_zscore(semantic)
            pred = predict_with_policy(
                classes,
                fused,
                [routes[i] for i in val_idx],
                [recalls[i] for i in val_idx],
                schema,
                scope="company",
                policy_name="exact_fallback_k5",
            )
            row = metric(val_labels, pred)
            key = f"top_k={top_k}:weight={weight}"
            search[key] = row
            if row["macro_f1"] > best_score:
                best_key = key
                best_score = row["macro_f1"]

    gain = round(best_score - baseline["macro_f1"], 6)
    payload: dict[str, object] = {
        "scope": "company",
        "protocol": (
            "production-like duplication-safe 80/20 pilot of article-to-article BGE triplet fine-tuning; "
            "hard negative labels mined only from fit-internal group-safe OOF confusion; no labeled-only subject fields"
        ),
        "baseline_validation": baseline,
        "search": search,
        "selected_key": best_key,
        "selected_validation_macro_f1": best_score,
        "validation_gain": gain,
        "pilot_gain_gate": PILOT_GAIN_GATE,
        "training": {
            "history": trained["history"],
            "training_seconds": trained["training_seconds"],
            "trainable_fraction": trained["trainable_fraction"],
            "peak_vram_mb": trained["peak_vram_mb"],
        },
        "external_touched": False,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
