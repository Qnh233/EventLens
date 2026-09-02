from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

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
from eventlens.exemplar_reranker import topk_exemplar_scores
from eventlens.io import read_competition_labeled_excel


TOP_K_VALUES = (1, 3, 5)
EXEMPLAR_WEIGHTS = (0.1, 0.25, 0.5)
MARGIN_QUANTILES = (0.1, 0.2, 0.3, 0.4)
OOF_GAIN_GATE = 0.005


def _row_zscore(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    return (matrix - mean) / np.maximum(std, 1e-6)


def _duplication_groups(articles) -> list[str]:
    return [
        str(row.duplication_id).strip()
        if row.duplication_id not in (None, "")
        else f"article:{row.article_id}"
        for row in articles
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in train]
    classes = sorted(set(labels))
    groups = _duplication_groups(train)
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
    train_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train
    ]

    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=settings.model.random_state,
    )
    svc_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    exemplar_oof = {
        top_k: np.full((len(train), len(classes)), -1e9, dtype=np.float64)
        for top_k in TOP_K_VALUES
    }
    leakage_checks: list[dict[str, int]] = []
    for fold, (fit_idx, val_idx) in enumerate(
        splitter.split(train_texts, labels, groups), start=1
    ):
        fit_groups = {groups[i] for i in fit_idx}
        val_groups = {groups[i] for i in val_idx}
        overlap = fit_groups & val_groups
        if overlap:
            raise RuntimeError("duplication group leaked across OOF fold")
        leakage_checks.append(
            {"fold": fold, "fit_groups": len(fit_groups), "val_groups": len(val_groups)}
        )
        svc = fit_with_descriptions(
            [train_texts[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        svc_oof[val_idx] = decision_scores(
            svc, [train_texts[i] for i in val_idx], classes
        )
        for top_k in TOP_K_VALUES:
            exemplar_oof[top_k][val_idx] = topk_exemplar_scores(
                np.asarray(train_vectors)[fit_idx],
                [labels[i] for i in fit_idx],
                np.asarray(train_vectors)[val_idx],
                classes,
                top_k=top_k,
            )

    svc_pred = predict_with_policy(
        classes,
        svc_oof,
        train_routes,
        train_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    baseline = metric(labels, svc_pred)
    search: dict[str, dict[str, float]] = {}
    best_key = "baseline"
    best_top_k = 0
    best_weight = 0.0
    best_mode = "baseline"
    best_threshold: float | None = None
    best_score = baseline["macro_f1"]
    svc_z = _row_zscore(svc_oof)
    for top_k in TOP_K_VALUES:
        exemplar_z = _row_zscore(exemplar_oof[top_k])
        for weight in EXEMPLAR_WEIGHTS:
            fused = svc_z + weight * exemplar_z
            pred = predict_with_policy(
                classes,
                fused,
                train_routes,
                train_recalls,
                schema,
                scope="company",
                policy_name="exact_fallback_k5",
            )
            row = metric(labels, pred)
            key = f"top_k={top_k}:weight={weight}"
            search[key] = row
            if row["macro_f1"] > best_score:
                best_key = key
                best_top_k = top_k
                best_weight = weight
                best_mode = "global"
                best_score = row["macro_f1"]

    selective_search: dict[str, dict[str, float | int]] = {}
    if best_top_k > 0:
        selected_scores = svc_z + best_weight * _row_zscore(exemplar_oof[best_top_k])
        selected_pred = predict_with_policy(
            classes,
            selected_scores,
            train_routes,
            train_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        order = np.argsort(-svc_oof, axis=1)
        margins = (
            svc_oof[np.arange(len(svc_oof)), order[:, 0]]
            - svc_oof[np.arange(len(svc_oof)), order[:, 1]]
        )
        svc_top1 = np.asarray([classes[int(index)] for index in order[:, 0]], dtype=object)
        bge_top1 = np.asarray(
            [row.candidates[0].event_name if row.candidates else "" for row in train_recalls],
            dtype=object,
        )
        disagreement = svc_top1 != bge_top1

        gate_specs: list[tuple[str, np.ndarray, float | None]] = [
            ("disagreement", disagreement, None)
        ]
        for quantile in MARGIN_QUANTILES:
            threshold = float(np.quantile(margins, quantile))
            gate_specs.append((f"margin_q{quantile}", margins <= threshold, threshold))
            gate_specs.append(
                (
                    f"disagreement_margin_q{quantile}",
                    disagreement & (margins <= threshold),
                    threshold,
                )
            )

        for name, mask, threshold in gate_specs:
            pred = list(svc_pred)
            for index in np.flatnonzero(mask):
                pred[int(index)] = selected_pred[int(index)]
            row = metric(labels, pred)
            selective_search[name] = {
                **row,
                "selected_count": int(mask.sum()),
                **({"threshold": round(threshold, 6)} if threshold is not None else {}),
            }
            if row["macro_f1"] > best_score:
                best_key = name
                best_mode = name
                best_threshold = threshold
                best_score = row["macro_f1"]

    gain = round(best_score - baseline["macro_f1"], 6)
    payload: dict[str, object] = {
        "scope": "company",
        "protocol": (
            "production-like duplication-safe 3-fold OOF multi-exemplar BGE reranking; "
            "same duplication_id never crosses folds; top-k/weight selected only on train OOF; "
            "external evaluated only if OOF Macro-F1 gain >= 0.005"
        ),
        "leakage_checks": leakage_checks,
        "baseline_group_safe_oof": baseline,
        "search": search,
        "selective_search": selective_search,
        "selected_key": best_key,
        "selected_exemplar": {
            "top_k": best_top_k,
            "weight": best_weight,
            "mode": best_mode,
            "margin_threshold": best_threshold,
        },
        "selected_oof_macro_f1": best_score,
        "oof_gain": gain,
        "external_gate_gain": OOF_GAIN_GATE,
        "external_touched": False,
        "production_like_external_reference_macro_f1": 0.770798,
    }

    if best_key != "baseline" and gain >= OOF_GAIN_GATE:
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
        test_texts = [
            routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test
        ]
        svc = fit_with_descriptions(
            train_texts, labels, desc_texts, desc_labels, repeat=1
        )
        svc_test = decision_scores(svc, test_texts, classes)
        exemplar_test = topk_exemplar_scores(
            train_vectors,
            labels,
            test_vectors,
            classes,
            top_k=best_top_k,
        )
        fused_test = _row_zscore(svc_test) + best_weight * _row_zscore(exemplar_test)
        fused_test_pred = predict_with_policy(
            classes,
            fused_test,
            test_routes,
            test_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        svc_test_pred = predict_with_policy(
            classes,
            svc_test,
            test_routes,
            test_recalls,
            schema,
            scope="company",
            policy_name="exact_fallback_k5",
        )
        if best_mode == "global":
            test_pred = fused_test_pred
        else:
            test_order = np.argsort(-svc_test, axis=1)
            test_margins = (
                svc_test[np.arange(len(svc_test)), test_order[:, 0]]
                - svc_test[np.arange(len(svc_test)), test_order[:, 1]]
            )
            test_svc_top1 = np.asarray(
                [classes[int(index)] for index in test_order[:, 0]], dtype=object
            )
            test_bge_top1 = np.asarray(
                [row.candidates[0].event_name if row.candidates else "" for row in test_recalls],
                dtype=object,
            )
            gate = np.ones(len(test), dtype=bool)
            if best_mode.startswith("disagreement"):
                gate &= test_svc_top1 != test_bge_top1
            if "margin_q" in best_mode:
                if best_threshold is None:
                    raise RuntimeError("selected margin gate is missing threshold")
                gate &= test_margins <= best_threshold
            test_pred = list(svc_test_pred)
            for index in np.flatnonzero(gate):
                test_pred[int(index)] = fused_test_pred[int(index)]
        external = metric(test_labels, test_pred)
        payload.update(
            {
                "external_touched": True,
                "external": external,
                "gain_vs_external_reference": round(
                    external["macro_f1"] - 0.770798, 6
                ),
                "gate_macro_f1": 0.80,
                "stretch_target_macro_f1": 0.85,
                "gate_passed": external["macro_f1"] >= 0.80,
            }
        )

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
