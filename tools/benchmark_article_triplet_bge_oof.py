from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_article_triplet_bge import (
    confusion_negative_map,
    duplication_groups,
    row_zscore,
)
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
from eventlens.challenge_evaluation import evaluate_challenge_slices
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.exemplar_reranker import topk_exemplar_scores
from eventlens.io import read_competition_labeled_excel


TOP_K = 3
FUSION_WEIGHT = 0.2
OOF_GAIN_GATE = 0.005


def should_touch_external(gain: float, *, allow_external: bool) -> bool:
    """External 只能在显式解锁且 OOF 过门禁时读取。"""
    return allow_external and gain >= OOF_GAIN_GATE


def resolve_random_state(cli_value: int | None, default_value: int) -> int:
    """允许 train-only 多 seed 稳定性复跑，不改变默认协议。"""
    return default_value if cli_value is None else cli_value


def metric(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def selective_masks(
    baseline_pred: list[str],
    routes,
    label_counts: Counter[str],
) -> dict[str, np.ndarray]:
    """仅使用推理时可见信号构造 selective expert 门控。"""

    ambiguous = np.asarray(
        [route.accepted_subject_code is None and len(route.candidates) > 1 for route in routes],
        dtype=bool,
    )
    rare_pred = np.asarray(
        [label_counts.get(label, 0) <= 20 for label in baseline_pred],
        dtype=bool,
    )
    return {
        "ambiguous_route": ambiguous,
        "rare_predicted_event": rare_pred,
        "ambiguous_or_rare": ambiguous | rare_pred,
    }


def apply_selective(
    baseline_pred: list[str],
    expert_pred: list[str],
    mask: np.ndarray,
) -> list[str]:
    if len(baseline_pred) != len(expert_pred) or len(mask) != len(baseline_pred):
        raise ValueError("selective prediction input length mismatch")
    return [
        expert if bool(use_expert) else baseline
        for baseline, expert, use_expert in zip(baseline_pred, expert_pred, mask)
    ]


def fit_hard_map(
    texts: list[str],
    labels: list[str],
    groups: list[str],
    classes: list[str],
    desc_texts: list[str],
    desc_labels: list[str],
    *,
    random_state: int,
) -> dict[int, list[int]]:
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=random_state)
    prediction = np.empty(len(labels), dtype=object)
    text_array = np.asarray(texts, dtype=object)
    label_array = np.asarray(labels, dtype=object)
    group_array = np.asarray(groups, dtype=object)
    for fit_idx, val_idx in splitter.split(text_array, label_array, group_array):
        model = fit_with_descriptions(
            text_array[fit_idx].tolist(),
            label_array[fit_idx].tolist(),
            desc_texts,
            desc_labels,
            repeat=1,
        )
        prediction[val_idx] = model.predict(text_array[val_idx].tolist())
    return confusion_negative_map(labels, prediction.tolist(), classes, top_k=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trainable-last-layers", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--selective-gating", action="store_true")
    parser.add_argument("--class-balanced-sampling", action="store_true")
    args = parser.parse_args()
    if args.trainable_last_layers <= 0:
        raise ValueError("trainable-last-layers must be positive")

    settings = load_settings()
    random_state = resolve_random_state(args.random_state, settings.model.random_state)
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in train]
    classes = sorted(set(labels))
    groups = duplication_groups(train)
    texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
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
    train_routes, train_recalls = build_routes_and_recalls(
        train,
        train_vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=client,
    )

    cfg = ArticleTripletTrainingConfig(
        model=args.model,
        max_length=384,
        max_content_chars=2400,
        batch_size=4,
        epochs=3,
        learning_rate=1e-5,
        margin=0.08,
        trainable_last_layers=args.trainable_last_layers,
        random_state=random_state,
        class_balanced_sampling=args.class_balanced_sampling,
    )
    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state,
    )
    svc_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    semantic_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    fold_reports: list[dict[str, object]] = []

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(texts, labels, groups), start=1):
        fit_groups = {groups[i] for i in fit_idx}
        val_groups = {groups[i] for i in val_idx}
        if fit_groups & val_groups:
            raise RuntimeError("duplication group leaked across OOF fold")
        fit_texts = [texts[i] for i in fit_idx]
        fit_labels = [labels[i] for i in fit_idx]
        fit_group_values = [groups[i] for i in fit_idx]

        svc = fit_with_descriptions(
            fit_texts,
            fit_labels,
            desc_texts,
            desc_labels,
            repeat=1,
        )
        svc_oof[val_idx] = decision_scores(svc, [texts[i] for i in val_idx], classes)
        hard_map = fit_hard_map(
            fit_texts,
            fit_labels,
            fit_group_values,
            classes,
            desc_texts,
            desc_labels,
            random_state=random_state,
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
        semantic_oof[val_idx] = topk_exemplar_scores(
            fit_vectors,
            fit_labels,
            val_vectors,
            classes,
            top_k=TOP_K,
        )
        fold_reports.append(
            {
                "fold": fold,
                "fit_count": len(fit_idx),
                "val_count": len(val_idx),
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

    baseline_pred = predict_with_policy(
        classes,
        svc_oof,
        train_routes,
        train_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    fused_oof = row_zscore(svc_oof) + FUSION_WEIGHT * row_zscore(semantic_oof)
    fused_pred = predict_with_policy(
        classes,
        fused_oof,
        train_routes,
        train_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    baseline = metric(labels, baseline_pred)
    selected = metric(labels, fused_pred)
    gain = round(selected["macro_f1"] - baseline["macro_f1"], 6)
    selected_mode = "global"
    selected_metric = selected
    selected_gain = gain
    selective_search: dict[str, dict[str, float | int]] = {}
    if args.selective_gating:
        label_counts = Counter(labels)
        for name, mask in selective_masks(baseline_pred, train_routes, label_counts).items():
            pred = apply_selective(baseline_pred, fused_pred, mask)
            score = metric(labels, pred)
            selective_search[name] = {
                **score,
                "selected_count": int(mask.sum()),
            }
            candidate_gain = round(score["macro_f1"] - baseline["macro_f1"], 6)
            if candidate_gain > selected_gain:
                selected_mode = name
                selected_metric = score
                selected_gain = candidate_gain
    challenge_kwargs = {
        "scope": "company",
        "rare_event_max_train_count": 20,
        "long_tail_source_max_train_count": 20,
        "long_text_percentile": 0.75,
    }
    baseline_challenge = evaluate_challenge_slices(
        train, train, baseline_pred, train_routes, **challenge_kwargs
    )
    triplet_challenge = evaluate_challenge_slices(
        train, train, fused_pred, train_routes, **challenge_kwargs
    )
    challenge_slice_gain = {
        name: round(triplet_challenge[name].macro_f1 - metrics.macro_f1, 6)
        for name, metrics in baseline_challenge.items()
    }
    payload: dict[str, object] = {
        "scope": "company",
        "protocol": (
            "production-like duplication-safe 3-fold OOF confirmation of fixed article-triplet BGE; "
            "all hyperparameters/top3/fusion0.2 frozen from prior train-only pilot; "
            "optional class-balanced anchor sampling uses sqrt inverse-frequency regularization; "
            "external gated on OOF gain"
        ),
        "fixed_config": {
            "epochs": 3,
            "trainable_last_layers": args.trainable_last_layers,
            "learning_rate": 1e-5,
            "margin": 0.08,
            "top_k": TOP_K,
            "fusion_weight": FUSION_WEIGHT,
            "random_state": random_state,
            "class_balanced_sampling": args.class_balanced_sampling,
            "class_balanced_sampling_strategy": (
                "sqrt_inverse_frequency" if args.class_balanced_sampling else "disabled"
            ),
        },
        "folds": fold_reports,
        "baseline_group_safe_oof": baseline,
        "triplet_fusion_oof": selected,
        "oof_gain": gain,
        "selective_gating_enabled": args.selective_gating,
        "selective_search": selective_search,
        "selected_mode": selected_mode,
        "selected_oof": selected_metric,
        "selected_oof_gain": selected_gain,
        "challenge_slices": {
            "protocol": (
                "train-only OOF diagnostic; labeled subject/source fields define slices only "
                "and are never classifier inputs"
            ),
            "baseline": {
                name: value.model_dump() for name, value in baseline_challenge.items()
            },
            "triplet": {
                name: value.model_dump() for name, value in triplet_challenge.items()
            },
            "macro_f1_gain": challenge_slice_gain,
        },
        "external_gate_gain": OOF_GAIN_GATE,
        "external_touched": False,
        "production_like_external_reference_macro_f1": 0.772131,
    }

    if should_touch_external(selected_gain, allow_external=args.allow_external):
        test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
        test_labels = [str(row.event_label) for row in test]
        test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
        if test_manifest.article_count != len(test):
            raise ValueError("test embedding count mismatch")
        if load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
            raise ValueError("test embedding order mismatch")
        test_routes, test_recalls = build_routes_and_recalls(
            test,
            test_vectors,
            scope="company",
            settings=settings,
            schema=schema,
            client=client,
        )
        final_svc = fit_with_descriptions(texts, labels, desc_texts, desc_labels, repeat=1)
        test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]
        test_svc = decision_scores(final_svc, test_texts, classes)
        full_hard_map = fit_hard_map(
            texts,
            labels,
            groups,
            classes,
            desc_texts,
            desc_labels,
            random_state=random_state,
        )
        final = train_article_triplet_encoder(
            train,
            labels,
            classes=classes,
            hard_negative_ids_by_class=full_hard_map,
            config=cfg,
            local_files_only=True,
        )
        final_train_vectors = encode_articles(
            final["model"],
            final["tokenizer"],
            train,
            max_length=cfg.max_length,
            max_content_chars=cfg.max_content_chars,
            batch_size=16,
        )
        final_test_vectors = encode_articles(
            final["model"],
            final["tokenizer"],
            test,
            max_length=cfg.max_length,
            max_content_chars=cfg.max_content_chars,
            batch_size=16,
        )
        test_semantic = topk_exemplar_scores(
            final_train_vectors,
            labels,
            final_test_vectors,
            classes,
            top_k=TOP_K,
        )
        test_fused = row_zscore(test_svc) + FUSION_WEIGHT * row_zscore(test_semantic)
        fused_test_pred = predict_with_policy(
            classes,
            test_fused,
            test_routes,
            test_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        baseline_test_pred = predict_with_policy(
            classes,
            test_svc,
            test_routes,
            test_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        if selected_mode == "global":
            test_pred = fused_test_pred
        else:
            masks = selective_masks(baseline_test_pred, test_routes, Counter(labels))
            test_pred = apply_selective(
                baseline_test_pred,
                fused_test_pred,
                masks[selected_mode],
            )
        external = metric(test_labels, test_pred)
        payload.update(
            {
                "external_touched": True,
                "external": external,
                "external_gain_vs_previous_best": round(external["macro_f1"] - 0.772131, 6),
                "final_training": {
                    "training_seconds": final["training_seconds"],
                    "peak_vram_mb": final["peak_vram_mb"],
                    "history": final["history"],
                },
            }
        )
    elif gain >= OOF_GAIN_GATE:
        payload["external_gate_status"] = (
            "OOF gate passed; external remains locked without --allow-external"
        )

    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
