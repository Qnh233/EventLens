from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from benchmark_cross_encoder_event import descriptions_for_scope
from benchmark_routed_subject_svc_v2 import routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.cross_encoder_event import build_article_text, build_event_text, choose_ranked_events
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.listwise_reranker import (
    build_label_to_article_pairs,
    build_candidate_groups,
    ensure_training_positive,
    freeze_reranker_last_layers,
    inverse_frequency_weights,
)


def metrics(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--last-layers", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))

    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if train_manifest.article_count != len(train) or test_manifest.article_count != len(test):
        raise ValueError("embedding count mismatch")
    if load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in train]:
        raise ValueError("train embedding order mismatch")
    if load_exported_article_ids(args.test_embeddings_dir) != [row.article_id for row in test]:
        raise ValueError("test embedding order mismatch")

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    descriptions = descriptions_for_scope(schema, "company")
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]

    native = settings.native_embedding
    embedding_client = NativeSentenceTransformerEmbeddingClient(
        model=native.model,
        device=native.device,
        batch_size=native.batch_size,
        normalize_embeddings=native.normalize_embeddings,
        cache_folder=native.cache_folder,
        local_files_only=True,
    )
    _, train_recalls = build_routes_and_recalls(
        train, train_vectors, scope="company", settings=settings, schema=schema, client=embedding_client
    )
    _, test_recalls = build_routes_and_recalls(
        test, test_vectors, scope="company", settings=settings, schema=schema, client=embedding_client
    )

    indices = np.arange(len(train))
    fit_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=settings.model.random_state,
        stratify=labels,
    )
    fit_idx = np.asarray(sorted(fit_idx), dtype=int)
    val_idx = np.asarray(sorted(val_idx), dtype=int)

    # Strict validation candidates: candidate classifier sees only fit labels.
    fit_svc = fit_with_descriptions(
        [train_texts[i] for i in fit_idx],
        [labels[i] for i in fit_idx],
        desc_texts,
        desc_labels,
        repeat=1,
    )
    val_svc_scores = decision_scores(fit_svc, [train_texts[i] for i in val_idx], classes)
    val_candidates_local = build_candidate_groups(
        val_svc_scores,
        classes,
        [train_recalls[i] for i in val_idx],
        svc_k=5,
        bge_k=5,
        limit=10,
    )

    # Training candidates are OOF within fit only, so validation labels cannot
    # influence candidate composition used by the reranker training objective.
    fit_labels = [labels[i] for i in fit_idx]
    fit_text_values = [train_texts[i] for i in fit_idx]
    fit_oof_scores = np.full((len(fit_idx), len(classes)), -1e9, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    for local_train, local_val in splitter.split(fit_text_values, fit_labels):
        svc = fit_with_descriptions(
            [fit_text_values[i] for i in local_train],
            [fit_labels[i] for i in local_train],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        fit_oof_scores[local_val] = decision_scores(
            svc,
            [fit_text_values[i] for i in local_val],
            classes,
        )
    fit_candidates = build_candidate_groups(
        fit_oof_scores,
        classes,
        [train_recalls[i] for i in fit_idx],
        svc_k=5,
        bge_k=5,
        limit=10,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model():
        model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
        trainable, total = freeze_reranker_last_layers(model, last_n=args.last_layers)
        model.to(device)
        return model, trainable, total

    class_weights = inverse_frequency_weights(fit_labels, power=0.5)

    def group_loss(model, article, truth, names, weight):
        names = ensure_training_positive(names, truth, limit=10)
        article_text = build_article_text(article, max_content_chars=1800)
        left, right = build_label_to_article_pairs(article_text, names, descriptions)
        encoded = tokenizer(
            left,
            right,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**encoded).logits.squeeze(-1).float()
            target = torch.tensor([names.index(truth)], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits[None, :], target)
        return loss * float(weight)

    def predict(model, articles, candidates):
        grouped_scores: list[list[float]] = []
        model.eval()
        with torch.inference_mode():
            for article, names in zip(articles, candidates):
                article_text = build_article_text(article, max_content_chars=1800)
                left, right = build_label_to_article_pairs(article_text, names, descriptions)
                encoded = tokenizer(
                    left,
                    right,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    scores = model(**encoded).logits.squeeze(-1).float()
                grouped_scores.append(scores.detach().cpu().tolist())
        return choose_ranked_events(candidates, grouped_scores)

    model, trainable, total = load_model()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    rng = np.random.default_rng(settings.model.random_state)
    history = []
    best_epoch = 1
    best_macro = -1.0
    best_state = None
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(fit_idx))
        running = 0.0
        for local_index in order:
            global_index = int(fit_idx[local_index])
            optimizer.zero_grad(set_to_none=True)
            loss = group_loss(
                model,
                train[global_index],
                labels[global_index],
                fit_candidates[local_index],
                class_weights[labels[global_index]],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            running += float(loss.detach().cpu())
        val_articles = [train[i] for i in val_idx]
        val_truth = [labels[i] for i in val_idx]
        val_pred = predict(model, val_articles, val_candidates_local)
        row = {
            "epoch": epoch,
            "train_loss": round(running / max(1, len(fit_idx)), 6),
            **metrics(val_truth, val_pred),
        }
        history.append(row)
        if row["macro_f1"] > best_macro:
            best_macro = row["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    validation_seconds = time.perf_counter() - started
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Final train OOF candidates, again with no self-label leakage.
    full_oof = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    for train_fold, val_fold in splitter.split(train_texts, labels):
        svc = fit_with_descriptions(
            [train_texts[i] for i in train_fold],
            [labels[i] for i in train_fold],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        full_oof[val_fold] = decision_scores(svc, [train_texts[i] for i in val_fold], classes)
    full_candidates = build_candidate_groups(full_oof, classes, train_recalls, limit=10)

    final_svc = fit_with_descriptions(train_texts, labels, desc_texts, desc_labels, repeat=1)
    test_svc_scores = decision_scores(final_svc, test_texts, classes)
    test_candidates = build_candidate_groups(test_svc_scores, classes, test_recalls, limit=10)
    test_candidate_hit = sum(label in names for label, names in zip(test_labels, test_candidates)) / len(test)

    final_model, final_trainable, final_total = load_model()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in final_model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    full_weights = inverse_frequency_weights(labels, power=0.5)
    rng = np.random.default_rng(settings.model.random_state)
    for _ in range(best_epoch):
        final_model.train()
        for index in rng.permutation(len(train)):
            optimizer.zero_grad(set_to_none=True)
            loss = group_loss(
                final_model,
                train[int(index)],
                labels[int(index)],
                full_candidates[int(index)],
                full_weights[labels[int(index)]],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in final_model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()

    test_pred = predict(final_model, test, test_candidates)
    external = metrics(test_labels, test_pred)
    payload = {
        "scope": "company",
        "protocol": "production-like listwise BGE reranker; strict fit-only validation candidates; train OOF candidate mining; inverse-sqrt tail weighting; only final reranker layers trainable; external used once after epoch selection",
        "model": args.model,
        "last_layers": args.last_layers,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "trainable_fraction": round(trainable / max(1, total), 6),
        "validation_history": history,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": round(best_macro, 6),
        "validation_seconds": round(validation_seconds, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "external_candidate_hit_rate": round(test_candidate_hit, 6),
        "external": external,
        "production_like_svc_reference_macro_f1": 0.770798,
        "gain_vs_svc": round(external["macro_f1"] - 0.770798, 6),
        "gate_macro_f1": 0.80,
        "stretch_target_macro_f1": 0.85,
        "gate_passed": external["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
