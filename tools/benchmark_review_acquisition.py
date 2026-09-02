from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_pretrained_reranker_cascade import build_candidate_sets, constrained
from benchmark_routed_subject_svc_v2 import metric, routed_text
from benchmark_schema_constrained_svc import (
    build_routes_and_recalls,
    decision_scores,
    fit_with_descriptions,
    label_description_examples,
)
from eventlens.config import load_settings
from eventlens.diverse_acquisition import diverse_balanced_round_robin
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex, NativeSentenceTransformerEmbeddingClient
from eventlens.io import read_competition_labeled_excel


def _selected_report(
    name: str,
    indices: np.ndarray,
    *,
    labels: list[str],
    predictions: list[str],
    candidates: list[list[str]],
    svc_top1: list[str],
    bge_top1: list[str],
    route_statuses: list[str],
) -> dict[str, object]:
    picked = indices.tolist()
    errors = sum(labels[i] != predictions[i] for i in picked)
    candidate_hits = sum(labels[i] in candidates[i] for i in picked)
    disagreements = sum(svc_top1[i] != bge_top1[i] for i in picked)
    candidate_only = sum(route_statuses[i] == "candidate_only" for i in picked)
    oracle_predictions = list(predictions)
    for i in picked:
        oracle_predictions[i] = labels[i]
    oracle = metric(labels, oracle_predictions)
    baseline = metric(labels, predictions)
    return {
        "strategy": name,
        "selected_count": len(picked),
        "error_count": int(errors),
        "error_rate": round(errors / max(1, len(picked)), 6),
        "candidate_truth_hit_rate": round(candidate_hits / max(1, len(picked)), 6),
        "svc_bge_disagreement_rate": round(disagreements / max(1, len(picked)), 6),
        "candidate_only_rate": round(candidate_only / max(1, len(picked)), 6),
        "oracle_macro_f1": oracle["macro_f1"],
        "oracle_macro_f1_gain": round(oracle["macro_f1"] - baseline["macro_f1"], 6),
    }


def _stable_top_k(keys: tuple[np.ndarray, ...], count: int) -> np.ndarray:
    """按 lexsort 的最后一个 key 为主键，稳定返回前 count 个索引。"""
    order = np.lexsort(keys)
    return order[:count]


def _balanced_round_robin(
    eligible: np.ndarray,
    group_keys: list[object],
    margins: np.ndarray,
    count: int,
) -> np.ndarray:
    """组内按 margin 排序后轮转抽样，避免复核预算被单一混淆模式占满。"""
    groups: dict[object, list[int]] = {}
    for index in eligible.tolist():
        groups.setdefault(group_keys[index], []).append(index)
    for values in groups.values():
        values.sort(key=lambda i: (float(margins[i]), i))

    ordered_groups = sorted(groups, key=lambda key: str(key))
    selected: list[int] = []
    offset = 0
    while len(selected) < count:
        added = False
        for key in ordered_groups:
            values = groups[key]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
        offset += 1
    return np.asarray(selected, dtype=np.int64)


def _parse_fractions(value: str | None, default: float) -> list[float]:
    if not value:
        return [default]
    fractions = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not fractions or any(not 0.0 < item < 1.0 for item in fractions):
        raise ValueError("fractions must contain values in (0, 1)")
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument(
        "--fractions",
        default=None,
        help="optional comma-separated review-budget frontier, e.g. 0.05,0.10,0.15,0.20",
    )
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    fractions = _parse_fractions(args.fractions, args.fraction)

    settings = load_settings()
    articles = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in articles]
    classes = sorted(set(labels))
    texts = [routed_text(row, None, mode="no_subject", max_content_chars=2400) for row in articles]

    manifest, vectors = load_exported_vectors(args.embeddings_dir)
    if manifest.article_count != len(articles):
        raise ValueError("embedding count mismatch")
    if load_exported_article_ids(args.embeddings_dir) != [row.article_id for row in articles]:
        raise ValueError("embedding order mismatch")

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
        articles, vectors, scope="company", settings=settings, schema=schema, client=client
    )

    scores = np.full((len(articles), len(classes)), -1e9, dtype=np.float64)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    for fit_idx, val_idx in folds.split(texts, labels):
        model = fit_with_descriptions(
            [texts[i] for i in fit_idx],
            [labels[i] for i in fit_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        scores[val_idx] = decision_scores(model, [texts[i] for i in val_idx], classes)

    predictions = constrained(classes, scores, routes, recalls, schema)
    order = np.argsort(-scores, axis=1)
    margins = scores[np.arange(len(scores)), order[:, 0]] - scores[np.arange(len(scores)), order[:, 1]]
    svc_top1 = [classes[int(row[0])] for row in order]
    bge_top1 = [row.candidates[0].event_name if row.candidates else "" for row in recalls]
    bge_margin = np.asarray(
        [
            (row.candidates[0].score - row.candidates[1].score)
            if len(row.candidates) >= 2
            else 1.0
            for row in recalls
        ],
        dtype=np.float64,
    )
    disagreement = np.asarray(
        [int(left != right) for left, right in zip(svc_top1, bge_top1)], dtype=np.int64
    )
    candidate_only = np.asarray(
        [int(not route.accepted_subject_code) for route in routes], dtype=np.int64
    )
    route_statuses = ["candidate_only" if value else "hard_route" for value in candidate_only]
    candidates = build_candidate_sets(scores, classes, recalls, svc_k=5, bge_k=5)
    selected_count = max(1, int(round(len(articles) * args.fraction)))

    challenge = settings.challenge_evaluation
    source_counts = Counter(row.source for row in articles if row.source)
    lengths = np.asarray(
        [len(row.title or "") + len(row.content or "") for row in articles],
        dtype=np.int64,
    )
    long_threshold = int(np.quantile(lengths, challenge.long_text_percentile))
    long_text = (lengths >= long_threshold).astype(np.int64)
    long_tail_source = np.asarray(
        [
            int(bool(row.source) and source_counts[row.source] <= challenge.long_tail_source_max_train_count)
            for row in articles
        ],
        dtype=np.int64,
    )
    observable_risk = long_text + long_tail_source + candidate_only
    disagreement_indices = np.flatnonzero(disagreement == 1)
    pair_keys = list(zip(svc_top1, bge_top1))
    duplicate_groups = [
        str(row.duplication_id).strip()
        if row.duplication_id not in (None, "")
        else f"article:{row.article_id}"
        for row in articles
    ]

    # 所有策略只依赖 production-like 推理时可见的不确定性/分歧信号，不使用 Gold 选样。
    strategies = {
        "low_margin": np.argsort(margins, kind="stable")[:selected_count],
        "disagreement_then_margin": _stable_top_k(
            (np.arange(len(articles)), margins, -disagreement), selected_count
        ),
        "candidate_only_then_margin": _stable_top_k(
            (np.arange(len(articles)), margins, -candidate_only), selected_count
        ),
        "disagreement_candidate_only_margin": _stable_top_k(
            (np.arange(len(articles)), margins, -candidate_only, -disagreement), selected_count
        ),
        "dual_low_margin": _stable_top_k(
            (np.arange(len(articles)), margins + bge_margin), selected_count
        ),
        "disagreement_risk_then_margin": _stable_top_k(
            (np.arange(len(articles)), margins, -observable_risk, -disagreement),
            selected_count,
        ),
        "risk_then_disagreement_margin": _stable_top_k(
            (np.arange(len(articles)), margins, -disagreement, -observable_risk),
            selected_count,
        ),
        "disagreement_class_balanced_margin": _balanced_round_robin(
            disagreement_indices, svc_top1, margins, selected_count
        ),
        "disagreement_pair_balanced_margin": _balanced_round_robin(
            disagreement_indices, pair_keys, margins, selected_count
        ),
        "disagreement_class_balanced_diverse_w025": diverse_balanced_round_robin(
            disagreement_indices,
            svc_top1,
            margins,
            vectors,
            selected_count,
            diversity_weight=0.25,
            duplicate_groups=duplicate_groups,
        ),
        "disagreement_class_balanced_diverse_w050": diverse_balanced_round_robin(
            disagreement_indices,
            svc_top1,
            margins,
            vectors,
            selected_count,
            diversity_weight=0.50,
            duplicate_groups=duplicate_groups,
        ),
    }
    reports = [
        _selected_report(
            name,
            indices,
            labels=labels,
            predictions=predictions,
            candidates=candidates,
            svc_top1=svc_top1,
            bge_top1=bge_top1,
            route_statuses=route_statuses,
        )
        for name, indices in strategies.items()
    ]

    frontier: list[dict[str, object]] = []
    diverse_frontier: list[dict[str, object]] = []
    for fraction in fractions:
        count = max(1, int(round(len(articles) * fraction)))
        class_balanced = _balanced_round_robin(
            disagreement_indices, svc_top1, margins, count
        )
        tranche = _selected_report(
            "disagreement_class_balanced_margin",
            class_balanced,
            labels=labels,
            predictions=predictions,
            candidates=candidates,
            svc_top1=svc_top1,
            bge_top1=bge_top1,
            route_statuses=route_statuses,
        )
        tranche["fraction"] = fraction
        frontier.append(tranche)
        for weight in (0.25, 0.50):
            diverse = diverse_balanced_round_robin(
                disagreement_indices,
                svc_top1,
                margins,
                vectors,
                count,
                diversity_weight=weight,
                duplicate_groups=duplicate_groups,
            )
            diverse_tranche = _selected_report(
                f"disagreement_class_balanced_diverse_w{int(weight * 100):03d}",
                diverse,
                labels=labels,
                predictions=predictions,
                candidates=candidates,
                svc_top1=svc_top1,
                bge_top1=bge_top1,
                route_statuses=route_statuses,
            )
            diverse_tranche["fraction"] = fraction
            diverse_tranche["diversity_weight"] = weight
            diverse_frontier.append(diverse_tranche)
    payload = {
        "scope": "company",
        "protocol": (
            "production-like company train 3-fold OOF acquisition benchmark; acquisition uses only "
            "inference-visible margin/disagreement/route signals; Gold is used only for aggregate offline evaluation"
        ),
        "baseline_oof": metric(labels, predictions),
        "fraction": args.fraction,
        "selected_count": selected_count,
        "fractions": fractions,
        "disagreement_class_balanced_frontier": frontier,
        "disagreement_class_balanced_diverse_frontier": diverse_frontier,
        "observable_risk_definition": {
            "long_text_threshold": long_threshold,
            "long_tail_source_max_train_count": challenge.long_tail_source_max_train_count,
            "signals": ["long_text", "long_tail_source", "candidate_only_route"],
        },
        "strategies": reports,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
