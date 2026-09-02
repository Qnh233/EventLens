from __future__ import annotations

import argparse
import json
import time
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
from benchmark_cross_encoder_event import descriptions_for_scope
from eventlens.config import load_settings
from eventlens.cross_encoder_event import build_article_text, build_event_text, merge_candidate_names
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel
from eventlens.schema_constrained_classifier import constrain_predictions
from eventlens.score_fusion import top1_margin


POOL_SPECS = ((5, 5), (8, 5))
RERANK_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
MARGIN_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5)
ORIENTATIONS = ("label_to_article", "article_to_label")


def build_candidate_sets(scores, classes, recalls, *, svc_k: int, bge_k: int):
    order = np.argsort(-scores, axis=1)
    output = []
    for index, recall in enumerate(recalls):
        svc = [classes[j] for j in order[index, :svc_k]]
        bge = [row.event_name for row in recall.candidates[:bge_k]]
        output.append(merge_candidate_names(svc, bge))
    return output


def candidate_rank_matrix(
    primary_scores: np.ndarray,
    reranker_scores: list[list[float]],
    candidates: list[list[str]],
    classes: list[str],
    *,
    reranker_weight: float,
) -> np.ndarray:
    if not 0.0 <= reranker_weight <= 1.0:
        raise ValueError("reranker_weight must be within [0, 1]")
    class_to_id = {label: index for index, label in enumerate(classes)}
    output = np.full_like(np.asarray(primary_scores, dtype=np.float64), -1e9)
    for row_index, (names, semantic) in enumerate(zip(candidates, reranker_scores)):
        if not names or len(names) != len(semantic):
            raise ValueError("candidate/reranker score mismatch")
        ids = [class_to_id[name] for name in names]
        local_primary = np.asarray(primary_scores[row_index, ids], dtype=np.float64)
        local_semantic = np.asarray(semantic, dtype=np.float64)
        primary_order = np.argsort(-local_primary)
        semantic_order = np.argsort(-local_semantic)
        primary_rank = np.empty(len(ids), dtype=np.float64)
        semantic_rank = np.empty(len(ids), dtype=np.float64)
        primary_rank[primary_order] = np.arange(len(ids), dtype=np.float64)
        semantic_rank[semantic_order] = np.arange(len(ids), dtype=np.float64)
        denom = float(max(1, len(ids) - 1))
        primary_rank = 1.0 - primary_rank / denom
        semantic_rank = 1.0 - semantic_rank / denom
        local = (1.0 - reranker_weight) * primary_rank + reranker_weight * semantic_rank
        output[row_index, ids] = local
    return output


def gated_matrix(
    primary_scores: np.ndarray,
    reranker_dense: np.ndarray,
    *,
    margin_threshold: float,
) -> np.ndarray:
    margins = top1_margin(primary_scores)
    output = np.asarray(primary_scores, dtype=np.float64).copy()
    mask = margins <= margin_threshold
    output[mask] = reranker_dense[mask]
    return output


def score_candidates(
    model,
    tokenizer,
    articles,
    candidates: list[list[str]],
    descriptions: dict[str, str],
    *,
    orientation: str,
    batch_size: int,
    max_length: int,
    max_content_chars: int,
    device,
    torch_module,
) -> list[list[float]]:
    pairs: list[tuple[str, str]] = []
    sizes: list[int] = []
    for article, names in zip(articles, candidates):
        article_text = build_article_text(article, max_content_chars=max_content_chars)
        sizes.append(len(names))
        for name in names:
            label_text = build_event_text(name, descriptions.get(name, ""))
            if orientation == "label_to_article":
                pairs.append((label_text, article_text))
            elif orientation == "article_to_label":
                pairs.append((article_text, label_text))
            else:
                raise ValueError(f"unsupported orientation: {orientation}")

    scores: list[float] = []
    model.eval()
    with torch_module.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = tokenizer(
                [left for left, _ in batch],
                [right for _, right in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded, return_dict=True).logits.reshape(-1).float()
            scores.extend(logits.detach().cpu().tolist())

    grouped: list[list[float]] = []
    offset = 0
    for size in sizes:
        grouped.append(scores[offset : offset + size])
        offset += size
    return grouped


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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-content-chars", type=int, default=2400)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    train_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=args.max_content_chars)
        for row in train
    ]
    test_texts = [
        routed_text(row, None, mode="no_subject", max_content_chars=args.max_content_chars)
        for row in test
    ]

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    descriptions = descriptions_for_scope(schema, "company")

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
    for fit_idx, val_idx in folds.split(train_texts, labels):
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
    svc_final = fit_with_descriptions(train_texts, labels, desc_texts, desc_labels, repeat=1)
    svc_test = decision_scores(svc_final, test_texts, classes)
    svc_oof_pred = constrained(classes, svc_oof, train_routes, train_recalls, schema)
    svc_external_pred = constrained(classes, svc_test, test_routes, test_recalls, schema)
    svc_oof_metric = metric(labels, svc_oof_pred)
    svc_external_metric = metric(test_labels, svc_external_pred)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        local_files_only=True,
    ).to(device)
    if device.type == "cuda":
        model.half()
        torch.cuda.reset_peak_memory_stats(device)

    search: dict[str, dict] = {}
    best = {
        "macro_f1": svc_oof_metric["macro_f1"],
        "key": "svc",
        "pool": None,
        "orientation": None,
        "family": "svc",
        "weight": 0.0,
        "threshold": None,
    }
    selected_train_candidates = None
    selected_train_reranker = None
    started = time.perf_counter()
    for svc_k, bge_k in POOL_SPECS:
        train_candidates = build_candidate_sets(
            svc_oof, classes, train_recalls, svc_k=svc_k, bge_k=bge_k
        )
        candidate_hit = sum(
            truth in names for truth, names in zip(labels, train_candidates)
        ) / len(train)
        for orientation in ORIENTATIONS:
            rerank = score_candidates(
                model,
                tokenizer,
                train,
                train_candidates,
                descriptions,
                orientation=orientation,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_content_chars=args.max_content_chars,
                device=device,
                torch_module=torch,
            )
            key_prefix = f"svc{svc_k}+bge{bge_k}:{orientation}"
            search[key_prefix] = {"candidate_hit_rate": round(candidate_hit, 6), "rank_blend": {}, "margin_gate": {}}
            for weight in RERANK_WEIGHTS:
                dense = candidate_rank_matrix(
                    svc_oof,
                    rerank,
                    train_candidates,
                    classes,
                    reranker_weight=weight,
                )
                pred = constrained(classes, dense, train_routes, train_recalls, schema)
                score = metric(labels, pred)
                key = f"w={weight}"
                search[key_prefix]["rank_blend"][key] = score
                if score["macro_f1"] > best["macro_f1"]:
                    best.update(
                        macro_f1=score["macro_f1"],
                        key=f"{key_prefix}:rank_blend:{key}",
                        pool=(svc_k, bge_k),
                        orientation=orientation,
                        family="rank_blend",
                        weight=weight,
                        threshold=None,
                    )
                    selected_train_candidates = train_candidates
                    selected_train_reranker = rerank

            reranker_only = candidate_rank_matrix(
                svc_oof,
                rerank,
                train_candidates,
                classes,
                reranker_weight=1.0,
            )
            margins = top1_margin(svc_oof)
            for quantile in MARGIN_QUANTILES:
                threshold = float(np.quantile(margins, quantile))
                dense = gated_matrix(
                    svc_oof,
                    reranker_only,
                    margin_threshold=threshold,
                )
                pred = constrained(classes, dense, train_routes, train_recalls, schema)
                score = metric(labels, pred)
                key = f"q={quantile}"
                search[key_prefix]["margin_gate"][key] = {
                    **score,
                    "threshold": round(threshold, 6),
                }
                if score["macro_f1"] > best["macro_f1"]:
                    best.update(
                        macro_f1=score["macro_f1"],
                        key=f"{key_prefix}:margin_gate:{key}",
                        pool=(svc_k, bge_k),
                        orientation=orientation,
                        family="margin_gate",
                        weight=1.0,
                        threshold=threshold,
                    )
                    selected_train_candidates = train_candidates
                    selected_train_reranker = rerank

    selection_seconds = time.perf_counter() - started

    if best["family"] == "svc":
        external_pred = svc_external_pred
        external_candidate_hit = None
    else:
        svc_k, bge_k = best["pool"]
        test_candidates = build_candidate_sets(
            svc_test, classes, test_recalls, svc_k=svc_k, bge_k=bge_k
        )
        external_candidate_hit = sum(
            truth in names for truth, names in zip(test_labels, test_candidates)
        ) / len(test)
        test_reranker = score_candidates(
            model,
            tokenizer,
            test,
            test_candidates,
            descriptions,
            orientation=best["orientation"],
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_content_chars=args.max_content_chars,
            device=device,
            torch_module=torch,
        )
        reranker_dense = candidate_rank_matrix(
            svc_test,
            test_reranker,
            test_candidates,
            classes,
            reranker_weight=best["weight"],
        )
        if best["family"] == "margin_gate":
            reranker_only = candidate_rank_matrix(
                svc_test,
                test_reranker,
                test_candidates,
                classes,
                reranker_weight=1.0,
            )
            final_scores = gated_matrix(
                svc_test,
                reranker_only,
                margin_threshold=float(best["threshold"]),
            )
        else:
            final_scores = reranker_dense
        external_pred = constrained(classes, final_scores, test_routes, test_recalls, schema)

    external = metric(test_labels, external_pred)
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    payload = {
        "scope": "company",
        "protocol": "production-like fixed pretrained BGE reranker cascade; 3-fold OOF char-SVC + BGE recall builds candidates and selects pool/orientation/fusion/gate; external touched once after selection",
        "model": args.model,
        "svc_oof": svc_oof_metric,
        "svc_external": svc_external_metric,
        "search": search,
        "selected": best,
        "selection_seconds": round(selection_seconds, 2),
        "peak_vram_mb": round(float(peak_vram), 1),
        "external_candidate_hit_rate": (
            round(float(external_candidate_hit), 6)
            if external_candidate_hit is not None
            else None
        ),
        "external": external,
        "gain_vs_svc": round(external["macro_f1"] - svc_external_metric["macro_f1"], 6),
        "gate_macro_f1": 0.80,
        "stretch_target_macro_f1": 0.85,
        "gate_passed": external["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
