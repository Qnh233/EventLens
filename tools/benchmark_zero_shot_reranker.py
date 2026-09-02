from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from benchmark_cross_encoder_event import build_candidate_sets, descriptions_for_scope
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
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    truths = [str(row.event_label) for row in train]
    test_truth = [str(row.event_label) for row in test]
    classes = sorted(set(truths))

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
    native = settings.native_embedding
    embedding_client = NativeSentenceTransformerEmbeddingClient(
        model=native.model,
        device=native.device,
        batch_size=native.batch_size,
        normalize_embeddings=native.normalize_embeddings,
        cache_folder=native.cache_folder,
        local_files_only=True,
    )
    _, test_recalls = build_routes_and_recalls(
        test,
        test_vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=embedding_client,
    )

    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in train]
    test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in test]
    svc = fit_with_descriptions(train_texts, truths, desc_texts, desc_labels, repeat=1)
    svc_scores = decision_scores(svc, test_texts, classes)
    candidates = build_candidate_sets(svc_scores, classes, test_recalls, svc_k=5, bge_k=5)
    candidate_hit = sum(label in names for label, names in zip(test_truth, candidates)) / len(test_truth)

    descriptions = descriptions_for_scope(schema, "company")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    pairs: list[tuple[str, str]] = []
    sizes: list[int] = []
    for article, names in zip(test, candidates):
        sizes.append(len(names))
        article_text = build_article_text(article, max_content_chars=1800)
        for name in names:
            pairs.append((article_text, build_event_text(name, descriptions.get(name, ""))))

    started = time.perf_counter()
    flat_scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            encoded = tokenizer(
                [row[0] for row in batch],
                [row[1] for row in batch],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.squeeze(-1)
            flat_scores.extend(logits.detach().float().cpu().tolist())
    prediction_seconds = time.perf_counter() - started

    grouped_scores: list[list[float]] = []
    offset = 0
    for size in sizes:
        grouped_scores.append(flat_scores[offset : offset + size])
        offset += size
    pred = choose_ranked_events(candidates, grouped_scores)
    external = metrics(test_truth, pred)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    payload = {
        "scope": "company",
        "protocol": "production-like zero-shot BGE reranker over fixed SVC Top5 union BGE Top5 candidates; no external hyperparameter selection; no labeled-only subject fields in article text",
        "model": args.model,
        "candidate_hit_rate": round(candidate_hit, 6),
        "pair_count": len(pairs),
        "external": external,
        "production_like_svc_reference_macro_f1": 0.770798,
        "gain_vs_svc": round(external["macro_f1"] - 0.770798, 6),
        "prediction_seconds": round(prediction_seconds, 3),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "gate_macro_f1": 0.80,
        "stretch_target_macro_f1": 0.85,
        "gate_passed": external["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
