from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_routed_subject_svc_v2 import metric, routed_text
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
from eventlens.label_anchor_contrastive import (
    LabelAnchorTrainingConfig,
    predict_label_anchor_scores,
    train_label_anchor_encoder,
)
from eventlens.schema_constrained_classifier import constrain_predictions
from eventlens.score_fusion import (
    blend_rank_scores,
    gated_predictions,
    top1_margin,
)


SEMANTIC_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
MARGIN_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5)


def constrained(classes, scores, routes, recalls, schema):
    return constrain_predictions(
        classes,
        scores,
        routes,
        recalls,
        schema,
        scope="company",
        policy="exact_subject_recall_fallback",
        recall_top_k=5,
    )


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
    train_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=2400)
        for row in train
    ]
    test_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=2400)
        for row in test
    ]

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    desc_by_label = dict(zip(desc_labels, desc_texts))
    anchor_texts = [desc_by_label[label] for label in classes]

    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding order mismatch")
    if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
        raise ValueError("test embedding order mismatch")
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

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    svc_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    anchor_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    fold_reports = []
    anchor_cfg = LabelAnchorTrainingConfig(
        model=args.model,
        max_length=512,
        max_content_chars=2400,
        batch_size=8,
        epochs=4,
        learning_rate=2e-5,
        class_weight_power=0.5,
        trainable_last_layers=2,
        temperature=0.07,
        random_state=settings.model.random_state,
    )
    for fold, (fit_idx, val_idx) in enumerate(folds.split(train_texts, labels), 1):
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
        trained = train_label_anchor_encoder(
            [train[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            label_texts=anchor_texts,
            label_names=classes,
            config=anchor_cfg,
            local_files_only=True,
        )
        anchor_oof[val_idx] = predict_label_anchor_scores(
            trained["model"],
            trained["tokenizer"],
            [train[i] for i in val_idx],
            anchor_vectors=trained["anchor_vectors"],
            config=anchor_cfg,
        )
        fold_reports.append(
            {
                "fold": fold,
                "anchor_training_seconds": trained["training_seconds"],
                "anchor_peak_vram_mb": trained["peak_vram_mb"],
            }
        )

    svc_oof_pred = constrained(classes, svc_oof, train_routes, train_recalls, schema)
    anchor_oof_pred = constrained(classes, anchor_oof, train_routes, train_recalls, schema)
    oof = {
        "svc": metric(labels, svc_oof_pred),
        "anchor": metric(labels, anchor_oof_pred),
        "rank_blend": {},
        "margin_gate": {},
    }
    best_family = "svc"
    best_key = "svc"
    best_score = oof["svc"]["macro_f1"]
    for weight in SEMANTIC_WEIGHTS:
        blended = blend_rank_scores(svc_oof, anchor_oof, semantic_weight=weight)
        pred = constrained(classes, blended, train_routes, train_recalls, schema)
        score = metric(labels, pred)
        key = f"semantic_weight={weight}"
        oof["rank_blend"][key] = score
        if score["macro_f1"] > best_score:
            best_family, best_key, best_score = "rank_blend", key, score["macro_f1"]

    margins = top1_margin(svc_oof)
    for quantile in MARGIN_QUANTILES:
        threshold = float(np.quantile(margins, quantile))
        pred = gated_predictions(
            svc_oof_pred,
            anchor_oof_pred,
            margins,
            margin_threshold=threshold,
        )
        score = metric(labels, pred)
        key = f"q={quantile}"
        oof["margin_gate"][key] = {**score, "threshold": round(threshold, 6)}
        if score["macro_f1"] > best_score:
            best_family, best_key, best_score = "margin_gate", key, score["macro_f1"]

    svc_final = fit_with_descriptions(
        train_texts, labels, desc_texts, desc_labels, repeat=1
    )
    svc_test = decision_scores(svc_final, test_texts, classes)
    anchor_final = train_label_anchor_encoder(
        train,
        labels,
        label_texts=anchor_texts,
        label_names=classes,
        config=anchor_cfg,
        local_files_only=True,
    )
    anchor_test = predict_label_anchor_scores(
        anchor_final["model"],
        anchor_final["tokenizer"],
        test,
        anchor_vectors=anchor_final["anchor_vectors"],
        config=anchor_cfg,
    )
    svc_test_pred = constrained(classes, svc_test, test_routes, test_recalls, schema)
    anchor_test_pred = constrained(classes, anchor_test, test_routes, test_recalls, schema)
    if best_family == "rank_blend":
        weight = float(best_key.split("=")[-1])
        final_scores = blend_rank_scores(svc_test, anchor_test, semantic_weight=weight)
        final_pred = constrained(classes, final_scores, test_routes, test_recalls, schema)
    elif best_family == "margin_gate":
        threshold = oof["margin_gate"][best_key]["threshold"]
        final_pred = gated_predictions(
            svc_test_pred,
            anchor_test_pred,
            top1_margin(svc_test),
            margin_threshold=threshold,
        )
    else:
        final_pred = svc_test_pred

    external = metric(test_labels, final_pred)
    payload = {
        "scope": "company",
        "protocol": "production-like 3-fold OOF fusion of no-subject char-SVC and label-anchor BGE; external used only after selecting global fusion family/weight or margin threshold",
        "folds": fold_reports,
        "oof": oof,
        "selected_family": best_family,
        "selected_key": best_key,
        "selected_oof_macro_f1": best_score,
        "external": external,
        "svc_external_reference": 0.770798,
        "macro_f1_gain_vs_svc": round(external["macro_f1"] - 0.770798, 6),
        "gate_macro_f1": 0.80,
        "gate_passed": external["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
