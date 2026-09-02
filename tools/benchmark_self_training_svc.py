from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

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
from eventlens.hard_examples import load_routed_recalls_jsonl
from eventlens.io import read_articles_excel, read_competition_labeled_excel
from eventlens.subject_routing import load_subject_routes_jsonl


def top1_and_margin(scores: np.ndarray, classes: list[str]) -> tuple[list[str], np.ndarray]:
    order = np.argsort(-scores, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    labels = [classes[index] for index in top1]
    margins = scores[np.arange(len(scores)), top1] - scores[np.arange(len(scores)), top2]
    return labels, margins


def calibrate_margin_threshold(
    truth: list[str],
    pred: list[str],
    margins: np.ndarray,
    *,
    target_precision: float,
    minimum_count: int,
) -> tuple[float, dict[str, float]]:
    candidates = sorted({float(value) for value in margins})
    best_threshold = float(max(candidates))
    best = {"count": 0, "precision": 0.0, "coverage": 0.0}
    for threshold in candidates:
        selected = [i for i, value in enumerate(margins) if value >= threshold]
        if len(selected) < minimum_count:
            continue
        correct = sum(pred[i] == truth[i] for i in selected)
        precision = correct / len(selected)
        if precision >= target_precision:
            best_threshold = threshold
            best = {
                "count": len(selected),
                "precision": round(precision, 6),
                "coverage": round(len(selected) / len(truth), 6),
            }
            break
    return best_threshold, best


PSEUDO_POLICIES = (
    "event_top1",
    "event_top3",
    "hard_event_top3",
    "exact_event_top3",
)


def structural_match(label: str, route, recall, *, policy: str) -> bool:
    top3 = [row.event_name for row in recall.candidates[:3]]
    if policy == "event_top1":
        return bool(top3) and top3[0] == label
    if policy == "event_top3":
        return label in top3
    if policy == "hard_event_top3":
        return route.accepted_subject_code is not None and label in top3
    if policy == "exact_event_top3":
        return route.method == "exact_alias" and label in top3
    raise ValueError(f"未知伪标签结构策略: {policy}")


def calibrate_pseudo_policy(
    truth: list[str],
    pred: list[str],
    margins: np.ndarray,
    routes,
    recalls,
    *,
    long_tail_labels: set[str],
    target_precision: float,
) -> tuple[str, float, dict[str, dict[str, float]]]:
    reports: dict[str, dict[str, float]] = {}
    best_policy = ""
    best_threshold = float("inf")
    best_count = -1
    for policy in PSEUDO_POLICIES:
        eligible = [
            i
            for i, label in enumerate(pred)
            if label in long_tail_labels
            and structural_match(label, routes[i], recalls[i], policy=policy)
        ]
        if not eligible:
            reports[policy] = {
                "eligible_count": 0,
                "selected_count": 0,
                "precision": 0.0,
                "coverage": 0.0,
                "threshold": 0.0,
            }
            continue
        threshold, calibration = calibrate_margin_threshold(
            [truth[i] for i in eligible],
            [pred[i] for i in eligible],
            np.asarray([margins[i] for i in eligible], dtype=np.float64),
            target_precision=target_precision,
            minimum_count=max(10, min(30, len(eligible) // 5)),
        )
        report = {
            "eligible_count": len(eligible),
            "selected_count": int(calibration["count"]),
            "precision": float(calibration["precision"]),
            "coverage": round(float(calibration["count"]) / max(1, len(truth)), 6),
            "threshold": round(float(threshold), 6),
        }
        reports[policy] = report
        if (
            report["precision"] >= target_precision
            and report["selected_count"] > best_count
        ):
            best_policy = policy
            best_threshold = threshold
            best_count = report["selected_count"]
    if not best_policy:
        raise RuntimeError("Gold OOF 未找到满足目标 precision 的伪标签策略")
    return best_policy, best_threshold, reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company"], default="company")
    parser.add_argument("--unlabeled-limit", type=int, default=50000)
    parser.add_argument("--routes", required=True)
    parser.add_argument("--recalls", required=True)
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--test-embeddings-dir", required=True)
    parser.add_argument("--max-pseudo-per-class", type=int, default=20)
    parser.add_argument("--target-oof-precision", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pseudo-output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)["company_event"]
    unlabeled = read_articles_excel(
        settings.paths.untagged_train,
        nrows=args.unlabeled_limit,
        task_scope="company_event",
    )
    routes = load_subject_routes_jsonl(args.routes, limit=len(unlabeled))
    recalls = load_routed_recalls_jsonl(args.recalls, limit=len(unlabeled))
    if not (len(unlabeled) == len(routes) == len(recalls)):
        raise ValueError("无标签文章/主体路由/事件召回数量不一致")
    ids = [row.article_id for row in unlabeled]
    if ids != [row.article_id for row in routes] or ids != [row.article_id for row in recalls]:
        raise ValueError("无标签文章与 route/recall 顺序不一致")

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    train_manifest, train_vectors = load_exported_vectors(args.train_embeddings_dir)
    if train_manifest.article_count != len(train) or load_exported_article_ids(args.train_embeddings_dir) != [
        row.article_id for row in train
    ]:
        raise ValueError("train embedding 顺序不一致")
    test_manifest, test_vectors = load_exported_vectors(args.test_embeddings_dir)
    if test_manifest.article_count != len(test) or load_exported_article_ids(args.test_embeddings_dir) != [
        row.article_id for row in test
    ]:
        raise ValueError("test embedding 顺序不一致")
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
    test_routes, test_recalls = build_routes_and_recalls(
        test,
        test_vectors,
        scope="company",
        settings=settings,
        schema=schema,
        client=client,
    )
    description_texts, description_labels = label_description_examples(schema, scope="company")
    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    classes = sorted(set(labels))
    max_chars = 2400
    train_texts = [routed_text(row, None, mode="no_subject", max_content_chars=max_chars) for row in train]
    test_texts = [routed_text(row, None, mode="no_subject", max_content_chars=max_chars) for row in test]
    unlabeled_texts = [
        routed_text(row, route, mode="no_subject", max_content_chars=max_chars)
        for row, route in zip(unlabeled, routes)
    ]

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    oof_scores = np.full((len(train), len(classes)), -1e9, dtype=np.float64)
    teacher_predictions: list[list[str]] = []
    teacher_margins: list[np.ndarray] = []
    for train_idx, val_idx in folds.split(train_texts, labels):
        model = fit_with_descriptions(
            [train_texts[i] for i in train_idx],
            [labels[i] for i in train_idx],
            description_texts,
            description_labels,
            repeat=1,
        )
        val_scores = decision_scores(model, [train_texts[i] for i in val_idx], classes)
        oof_scores[val_idx] = val_scores
        unlabeled_scores = decision_scores(model, unlabeled_texts, classes)
        pred, margins = top1_and_margin(unlabeled_scores, classes)
        teacher_predictions.append(pred)
        teacher_margins.append(margins)

    train_counts = Counter(labels)
    sorted_counts = sorted(train_counts.values())
    median_count = float(np.median(sorted_counts))
    long_tail_labels = {label for label, count in train_counts.items() if count <= median_count}
    oof_pred, oof_margins = top1_and_margin(oof_scores, classes)
    selected_policy, margin_threshold, policy_calibration = calibrate_pseudo_policy(
        labels,
        oof_pred,
        oof_margins,
        train_routes,
        train_recalls,
        long_tail_labels=long_tail_labels,
        target_precision=args.target_oof_precision,
    )
    candidates: list[dict] = []
    reject_reasons = Counter()
    for index, (article, route, recall) in enumerate(zip(unlabeled, routes, recalls)):
        labels_here = [teacher[index] for teacher in teacher_predictions]
        if len(set(labels_here)) != 1:
            reject_reasons["teacher_disagreement"] += 1
            continue
        label = labels_here[0]
        if label not in long_tail_labels:
            reject_reasons["not_long_tail"] += 1
            continue
        min_margin = min(float(margins[index]) for margins in teacher_margins)
        if min_margin < margin_threshold:
            reject_reasons["margin_below_gate"] += 1
            continue
        if not structural_match(label, route, recall, policy=selected_policy):
            reject_reasons["structure_gate_failed"] += 1
            continue
        candidates.append(
            {
                "index": index,
                "article_id": article.article_id,
                "event_label": label,
                "teacher_min_margin": round(min_margin, 6),
                "subject_code": route.accepted_subject_code
                or (route.candidates[0].subject_code if route.candidates else None),
                "subject_method": route.method,
                "event_recall_top1_score": recall.candidates[0].score if recall.candidates else None,
                "pseudo_policy": selected_policy,
            }
        )

    candidates.sort(key=lambda row: (-row["teacher_min_margin"], row["article_id"]))
    per_class = Counter()
    selected: list[dict] = []
    for row in candidates:
        label = row["event_label"]
        if per_class[label] >= args.max_pseudo_per_class:
            continue
        selected.append(row)
        per_class[label] += 1

    selected_indices = [row["index"] for row in selected]
    pseudo_texts = [unlabeled_texts[index] for index in selected_indices]
    pseudo_labels = [row["event_label"] for row in selected]

    base_model = fit_with_descriptions(
        train_texts,
        labels,
        description_texts,
        description_labels,
        repeat=1,
    )
    base_scores = decision_scores(base_model, test_texts, classes)
    base_pred = predict_with_policy(
        classes,
        base_scores,
        test_routes,
        test_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )

    augmented_model = fit_with_descriptions(
        train_texts + pseudo_texts,
        labels + pseudo_labels,
        description_texts,
        description_labels,
        repeat=1,
    )
    augmented_scores = decision_scores(augmented_model, test_texts, classes)
    augmented_pred = predict_with_policy(
        classes,
        augmented_scores,
        test_routes,
        test_recalls,
        schema,
        scope="company",
        policy_name="exact_fallback_k5",
    )
    base_metrics = metric(test_labels, base_pred)
    augmented_metrics = metric(test_labels, augmented_pred)

    pseudo_path = Path(args.pseudo_output)
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    with pseudo_path.open("w", encoding="utf-8") as file:
        for row in selected:
            payload = dict(row)
            payload.pop("index", None)
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    payload = {
        "scope": "company",
        "protocol": "target-domain pilot; Gold OOF selects >= target precision structural gate; 3-fold teacher agreement + calibrated margin + long-tail only",
        "unlabeled_count": len(unlabeled),
        "selected_pseudo_policy": selected_policy,
        "pseudo_policy_calibration": policy_calibration,
        "oof_margin_gate": {
            "target_precision": args.target_oof_precision,
            "threshold": round(margin_threshold, 6),
            **policy_calibration[selected_policy],
        },
        "long_tail_train_count_median": median_count,
        "long_tail_label_count": len(long_tail_labels),
        "candidate_count": len(candidates),
        "selected_pseudo_count": len(selected),
        "selected_per_class": dict(sorted(per_class.items())),
        "reject_reasons": dict(reject_reasons),
        "base_external": base_metrics,
        "augmented_external": augmented_metrics,
        "macro_f1_gain": round(augmented_metrics["macro_f1"] - base_metrics["macro_f1"], 6),
        "gate_macro_f1": 0.80,
        "gate_passed": augmented_metrics["macro_f1"] >= 0.80,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
