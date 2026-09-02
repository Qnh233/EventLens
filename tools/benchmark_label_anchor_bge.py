from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
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


def metric(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def predictions_from_scores(
    classes,
    scores,
    routes,
    recalls,
    schema,
    *,
    policy: str,
):
    if policy == "global":
        return [classes[index] for index in np.argmax(scores, axis=1)]
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
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    description_texts, description_labels = label_description_examples(schema, scope="company")
    description_by_label = dict(zip(description_labels, description_texts))
    anchor_texts = [description_by_label[label] for label in classes]

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

    indices = np.arange(len(train))
    fit_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=settings.model.random_state,
        stratify=labels,
    )
    candidate_configs = [
        {"last_layers": 1, "lr": 1e-5, "weight_power": 0.0},
        {"last_layers": 1, "lr": 2e-5, "weight_power": 0.5},
        {"last_layers": 2, "lr": 1e-5, "weight_power": 0.0},
        {"last_layers": 2, "lr": 1e-5, "weight_power": 0.5},
        {"last_layers": 2, "lr": 2e-5, "weight_power": 0.5},
    ]
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    validation: dict[str, dict] = {}
    best_key = ""
    best_macro = -1.0
    best_spec = None
    best_policy = "global"
    best_epoch = 4
    for spec in candidate_configs:
        key = f"last{spec['last_layers']}:lr{spec['lr']}:w{spec['weight_power']}"
        cfg = LabelAnchorTrainingConfig(
            model=args.model,
            max_length=512,
            max_content_chars=2400,
            batch_size=8,
            epochs=4,
            learning_rate=spec["lr"],
            class_weight_power=spec["weight_power"],
            trainable_last_layers=spec["last_layers"],
            temperature=0.07,
            random_state=settings.model.random_state,
        )
        trained = train_label_anchor_encoder(
            [train[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            label_texts=anchor_texts,
            label_names=classes,
            config=cfg,
            local_files_only=True,
        )
        scores = predict_label_anchor_scores(
            trained["model"],
            trained["tokenizer"],
            [train[i] for i in val_idx],
            anchor_vectors=trained["anchor_vectors"],
            config=cfg,
        )
        policy_scores = {}
        for policy in ("global", "exact_fallback_k5"):
            pred = predictions_from_scores(
                classes,
                scores,
                [train_routes[i] for i in val_idx],
                [train_recalls[i] for i in val_idx],
                schema,
                policy=policy,
            )
            policy_scores[policy] = metric([labels[i] for i in val_idx], pred)
            if policy_scores[policy]["macro_f1"] > best_macro:
                best_macro = policy_scores[policy]["macro_f1"]
                best_key = key
                best_spec = dict(spec)
                best_policy = policy
        validation[key] = {
            "metrics": policy_scores,
            "trainable_fraction": trained["trainable_fraction"],
            "training_seconds": trained["training_seconds"],
            "peak_vram_mb": trained["peak_vram_mb"],
            "history": trained["history"],
        }
        del trained

    if best_spec is None:
        raise RuntimeError("no label-anchor candidate completed")
    final_cfg = LabelAnchorTrainingConfig(
        model=args.model,
        max_length=512,
        max_content_chars=2400,
        batch_size=8,
        epochs=best_epoch,
        learning_rate=best_spec["lr"],
        class_weight_power=best_spec["weight_power"],
        trainable_last_layers=best_spec["last_layers"],
        temperature=0.07,
        random_state=settings.model.random_state,
    )
    final_model_dir = work_dir / "selected"
    if final_model_dir.exists():
        shutil.rmtree(final_model_dir)
    final = train_label_anchor_encoder(
        train,
        labels,
        label_texts=anchor_texts,
        label_names=classes,
        config=final_cfg,
        output_dir=final_model_dir,
        local_files_only=True,
    )
    test_scores = predict_label_anchor_scores(
        final["model"],
        final["tokenizer"],
        test,
        anchor_vectors=final["anchor_vectors"],
        config=final_cfg,
    )
    test_pred = predictions_from_scores(
        classes,
        test_scores,
        test_routes,
        test_recalls,
        schema,
        policy=best_policy,
    )
    external = metric(test_labels, test_pred)
    payload = {
        "scope": "company",
        "protocol": "production-like label-anchored BGE-M3; no true subject fields in encoder text; 80/20 stratified train validation selects encoder depth/lr/class balance/policy; external used once",
        "selected_key": best_key,
        "selected_spec": best_spec,
        "selected_policy": best_policy,
        "selected_validation": validation[best_key]["metrics"][best_policy],
        "validation": validation,
        "final_training": {
            "trainable_fraction": final["trainable_fraction"],
            "training_seconds": final["training_seconds"],
            "peak_vram_mb": final["peak_vram_mb"],
            "history": final["history"],
        },
        "external": external,
        "production_like_svc_reference_macro_f1": 0.770798,
        "macro_f1_gain_vs_svc": round(external["macro_f1"] - 0.770798, 6),
        "gate_macro_f1": 0.80,
        "gate_passed": external["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
