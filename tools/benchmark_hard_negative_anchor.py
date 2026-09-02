from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from benchmark_label_anchor_bge import predictions_from_scores
from benchmark_schema_constrained_svc import (
    build_model,
    build_routes_and_recalls,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.hard_negative_mining import build_confusion_hard_negative_map
from eventlens.io import read_competition_labeled_excel
from eventlens.label_anchor_contrastive import (
    LabelAnchorTrainingConfig,
    build_production_text,
    predict_label_anchor_scores,
    train_label_anchor_encoder,
)


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
    anchor_vectors = np.asarray(client.embed(anchor_texts), dtype=np.float32)
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
    fit_articles = [train[i] for i in fit_idx]
    fit_labels = [labels[i] for i in fit_idx]
    fit_texts = [build_production_text(row, max_content_chars=2400) for row in fit_articles]

    candidate_specs = [
        {"top_k": 3, "semantic_weight": 0.35, "hard_weight": 0.0, "margin": 0.1},
        {"top_k": 2, "semantic_weight": 0.25, "hard_weight": 0.1, "margin": 0.05},
        {"top_k": 2, "semantic_weight": 0.35, "hard_weight": 0.25, "margin": 0.05},
        {"top_k": 3, "semantic_weight": 0.35, "hard_weight": 0.1, "margin": 0.1},
        {"top_k": 3, "semantic_weight": 0.50, "hard_weight": 0.25, "margin": 0.1},
        {"top_k": 3, "semantic_weight": 0.35, "hard_weight": 0.5, "margin": 0.1},
        {"top_k": 4, "semantic_weight": 0.35, "hard_weight": 0.25, "margin": 0.1},
    ]
    validation: dict[str, dict] = {}
    best_key = ""
    best_spec = None
    best_macro = -1.0
    for spec in candidate_specs:
        hard_map = build_confusion_hard_negative_map(
            fit_texts,
            fit_labels,
            classes,
            anchor_vectors,
            model_factory=build_model,
            top_k=spec["top_k"],
            semantic_weight=spec["semantic_weight"],
            random_state=settings.model.random_state,
        )
        cfg = LabelAnchorTrainingConfig(
            model=args.model,
            max_length=512,
            max_content_chars=2400,
            batch_size=8,
            epochs=4,
            learning_rate=2e-5,
            class_weight_power=0.5,
            trainable_last_layers=2,
            temperature=0.07,
            hard_negative_weight=spec["hard_weight"],
            hard_negative_margin=spec["margin"],
            random_state=settings.model.random_state,
        )
        trained = train_label_anchor_encoder(
            fit_articles,
            fit_labels,
            label_texts=anchor_texts,
            label_names=classes,
            config=cfg,
            hard_negative_ids_by_class=hard_map if spec["hard_weight"] > 0 else None,
            local_files_only=True,
        )
        scores = predict_label_anchor_scores(
            trained["model"],
            trained["tokenizer"],
            [train[i] for i in val_idx],
            anchor_vectors=trained["anchor_vectors"],
            config=cfg,
        )
        pred = predictions_from_scores(
            classes,
            scores,
            [train_routes[i] for i in val_idx],
            [train_recalls[i] for i in val_idx],
            schema,
            policy="exact_fallback_k5",
        )
        metrics = metric([labels[i] for i in val_idx], pred)
        key = (
            f"k{spec['top_k']}:sem{spec['semantic_weight']}:"
            f"w{spec['hard_weight']}:m{spec['margin']}"
        )
        validation[key] = {
            "spec": spec,
            "metrics": metrics,
            "training_seconds": trained["training_seconds"],
            "peak_vram_mb": trained["peak_vram_mb"],
            "history": trained["history"],
        }
        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            best_key = key
            best_spec = dict(spec)
        del trained

    if best_spec is None:
        raise RuntimeError("no hard-negative candidate completed")

    full_texts = [build_production_text(row, max_content_chars=2400) for row in train]
    full_hard_map = build_confusion_hard_negative_map(
        full_texts,
        labels,
        classes,
        anchor_vectors,
        model_factory=build_model,
        top_k=best_spec["top_k"],
        semantic_weight=best_spec["semantic_weight"],
        random_state=settings.model.random_state,
    )
    final_cfg = LabelAnchorTrainingConfig(
        model=args.model,
        max_length=512,
        max_content_chars=2400,
        batch_size=8,
        epochs=4,
        learning_rate=2e-5,
        class_weight_power=0.5,
        trainable_last_layers=2,
        temperature=0.07,
        hard_negative_weight=best_spec["hard_weight"],
        hard_negative_margin=best_spec["margin"],
        random_state=settings.model.random_state,
    )
    final = train_label_anchor_encoder(
        train,
        labels,
        label_texts=anchor_texts,
        label_names=classes,
        config=final_cfg,
        hard_negative_ids_by_class=full_hard_map if best_spec["hard_weight"] > 0 else None,
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
        policy="exact_fallback_k5",
    )
    external = metric(test_labels, test_pred)
    payload = {
        "scope": "company",
        "protocol": "production-like confusion-aware hard-negative label-anchor BGE; hard negatives mined from train-only 3-fold SVC confusion + Schema label semantics; 80/20 train validation selects loss settings; external used once",
        "selected_key": best_key,
        "selected_spec": best_spec,
        "selected_validation": validation[best_key]["metrics"],
        "validation": validation,
        "external": external,
        "label_anchor_reference_external": 0.712678,
        "production_like_svc_reference_macro_f1": 0.770798,
        "gain_vs_anchor": round(external["macro_f1"] - 0.712678, 6),
        "gain_vs_svc": round(external["macro_f1"] - 0.770798, 6),
        "gate_macro_f1": 0.80,
        "stretch_target_macro_f1": 0.85,
        "gate_passed": external["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
