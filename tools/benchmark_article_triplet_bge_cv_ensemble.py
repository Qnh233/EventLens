from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_article_triplet_bge import duplication_groups, row_zscore
from benchmark_article_triplet_bge_oof import fit_hard_map
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


TOP_K = 3
FUSION_WEIGHT = 0.2


def metric(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
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
    groups = duplication_groups(train)
    texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]

    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train):
        raise ValueError("train embedding count mismatch")
    if test_manifest.article_count != len(test):
        raise ValueError("test embedding count mismatch")
    if load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding order mismatch")
    if load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
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
    test_routes, test_recalls = build_routes_and_recalls(
        test,
        test_vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=client,
    )

    full_svc = fit_with_descriptions(texts, labels, desc_texts, desc_labels, repeat=1)
    test_svc = decision_scores(full_svc, test_texts, classes)
    baseline_pred = predict_with_policy(
        classes,
        test_svc,
        test_routes,
        test_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )

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
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=settings.model.random_state,
    )
    semantic_scores: list[np.ndarray] = []
    fold_reports: list[dict[str, object]] = []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(texts, labels, groups), start=1):
        fit_groups = {groups[i] for i in fit_idx}
        val_groups = {groups[i] for i in val_idx}
        if fit_groups & val_groups:
            raise RuntimeError("duplication group leaked across CV fold")
        fit_texts = [texts[i] for i in fit_idx]
        fit_labels = [labels[i] for i in fit_idx]
        fit_group_values = [groups[i] for i in fit_idx]
        hard_map = fit_hard_map(
            fit_texts,
            fit_labels,
            fit_group_values,
            classes,
            desc_texts,
            desc_labels,
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
        fold_test_vectors = encode_articles(
            trained["model"],
            trained["tokenizer"],
            test,
            max_length=cfg.max_length,
            max_content_chars=cfg.max_content_chars,
            batch_size=16,
        )
        semantic_scores.append(
            topk_exemplar_scores(
                fit_vectors,
                fit_labels,
                fold_test_vectors,
                classes,
                top_k=TOP_K,
            )
        )
        fold_reports.append(
            {
                "fold": fold,
                "fit_count": len(fit_idx),
                "heldout_count": len(val_idx),
                "overlap_groups": 0,
                "training_seconds": trained["training_seconds"],
                "peak_vram_mb": trained["peak_vram_mb"],
                "history": trained["history"],
            }
        )
        del trained
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    # 每个 fold 的 semantic score 先行内标准化再平均，避免尺度漂移。
    semantic_ensemble = np.mean([row_zscore(score) for score in semantic_scores], axis=0)
    fused = row_zscore(test_svc) + FUSION_WEIGHT * semantic_ensemble
    pred = predict_with_policy(
        classes,
        fused,
        test_routes,
        test_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    external = metric(test_labels, pred)
    baseline_external = metric(test_labels, baseline_pred)
    payload = {
        "scope": "company",
        "protocol": (
            "production-like fixed 3-fold CV ensemble of article-triplet BGE models; "
            "same frozen configuration confirmed by prior group-safe OOF; external labels used only for final metric"
        ),
        "fixed_config": {
            "epochs": 3,
            "trainable_last_layers": 2,
            "learning_rate": 1e-5,
            "margin": 0.08,
            "top_k": TOP_K,
            "fusion_weight": FUSION_WEIGHT,
        },
        "folds": fold_reports,
        "svc_external": baseline_external,
        "cv_ensemble_external": external,
        "gain_vs_svc": round(external["macro_f1"] - baseline_external["macro_f1"], 6),
        "previous_best_external_macro_f1": 0.772131,
        "gain_vs_previous_best": round(external["macro_f1"] - 0.772131, 6),
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
