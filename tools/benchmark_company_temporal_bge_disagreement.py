from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_article_ids, load_exported_vectors
from eventlens.event_retrieval import EventSchemaIndex
from eventlens.exemplar_reranker import topk_exemplar_scores
from eventlens.io import read_competition_labeled_excel
from tools.benchmark_company_temporal_holdout import (
    _disagreement_class_balanced_indices,
    _predicted_class_balanced_temporal_coverage_indices,
    paired_bootstrap_macro_f1_gain,
    production_like_text,
    temporal_three_way_group_split_at_end,
)
from tools.benchmark_schema_constrained_svc import fit_with_descriptions, label_description_examples


REVIEW_BUDGET_FRACTION = 0.20
HYBRID_DISAGREEMENT_SHARE = 0.50
END_FRACTIONS = (0.65, 0.75, 0.85, 0.95, 1.0)


def _top1_exemplar_prediction(
    history_vectors: np.ndarray,
    history_labels: list[str],
    query_vectors: np.ndarray,
    classes: list[str],
) -> list[str]:
    scores = topk_exemplar_scores(
        history_vectors,
        history_labels,
        query_vectors,
        classes,
        top_k=1,
    )
    return [classes[index] for index in np.asarray(scores).argmax(axis=1)]


def _hybrid_disagreement_temporal_indices(
    primary_pred: list[str],
    secondary_pred: list[str],
    margins: np.ndarray,
    publish_times,
    *,
    count: int,
) -> tuple[list[int], int, int]:
    """固定 50/50：一半预算投向异构 disagreement，一半补时间覆盖。"""
    if count <= 0:
        return [], 0, 0
    disagreement_budget = int(round(count * HYBRID_DISAGREEMENT_SHARE))
    disagreement_selected = _disagreement_class_balanced_indices(
        primary_pred,
        secondary_pred,
        margins,
        disagreement_budget,
    )
    selected = list(disagreement_selected)
    selected_set = set(selected)

    remaining = count - len(selected)
    if remaining > 0:
        coverage_order = _predicted_class_balanced_temporal_coverage_indices(
            primary_pred,
            publish_times,
            count=len(primary_pred),
        )
        for index in coverage_order:
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            if len(selected) >= count:
                break
    return selected, len(disagreement_selected), len(selected) - len(disagreement_selected)


def rolling_svc_bge_disagreement_experiment(
    rows,
    vectors: np.ndarray,
    schema,
    *,
    end_fraction: float,
) -> dict:
    """固定20% review：SVC 与 frozen BGE Top1 Gold exemplar 分歧后按预测类均衡选样。"""
    (
        history_idx,
        review_idx,
        future_idx,
        dropped_idx,
        review_cutoff,
        future_cutoff,
        end_time,
    ) = temporal_three_way_group_split_at_end(
        rows,
        end_fraction=end_fraction,
        review_ratio=0.15,
        future_ratio=0.15,
    )
    history_texts = [production_like_text(rows[i]) for i in history_idx]
    history_labels = [str(rows[i].event_label) for i in history_idx]
    review_texts = [production_like_text(rows[i]) for i in review_idx]
    review_labels = [str(rows[i].event_label) for i in review_idx]
    future_texts = [production_like_text(rows[i]) for i in future_idx]
    future_labels = [str(rows[i].event_label) for i in future_idx]
    classes = sorted(set(history_labels))
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    primary = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    primary_review_pred = primary.predict(review_texts).tolist()
    review_scores = np.asarray(primary.decision_function(review_texts), dtype=np.float64)
    order = np.argsort(-review_scores, axis=1)
    margins = review_scores[np.arange(len(review_scores)), order[:, 0]] - review_scores[
        np.arange(len(review_scores)), order[:, 1]
    ]

    secondary_review_pred = _top1_exemplar_prediction(
        np.asarray(vectors)[history_idx],
        history_labels,
        np.asarray(vectors)[review_idx],
        classes,
    )
    budget = max(1, int(round(len(review_idx) * REVIEW_BUDGET_FRACTION)))
    selected_local = _disagreement_class_balanced_indices(
        primary_review_pred,
        secondary_review_pred,
        margins,
        budget,
    )
    selected_texts = [review_texts[i] for i in selected_local]
    selected_labels = [review_labels[i] for i in selected_local]
    selected_errors = sum(primary_review_pred[i] != review_labels[i] for i in selected_local)

    baseline_future_pred = primary.predict(future_texts).tolist()
    baseline_future_f1 = f1_score(future_labels, baseline_future_pred, average="macro", zero_division=0)
    retrained = fit_with_descriptions(
        history_texts + selected_texts,
        history_labels + selected_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    retrained_future_pred = retrained.predict(future_texts).tolist()
    retrained_future_f1 = f1_score(
        future_labels,
        retrained_future_pred,
        average="macro",
        zero_division=0,
    )
    disagreement_count = sum(
        left != right for left, right in zip(primary_review_pred, secondary_review_pred)
    )
    return {
        "protocol": "train_only_svc_vs_frozen_bge_top1_disagreement_class_balanced_review",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "review_window_count": len(review_idx),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "review_budget_fraction": REVIEW_BUDGET_FRACTION,
        "requested_review_count": budget,
        "disagreement_count": disagreement_count,
        "selected_count": len(selected_local),
        "selection": "svc_vs_frozen_bge_top1_disagreement_then_svc_predicted_class_balance_and_margin",
        "selection_uses_review_gold": False,
        "selection_uses_future_gold": False,
        "selected_primary_error_rate": round(selected_errors / max(1, len(selected_local)), 6),
        "baseline_future_macro_f1": round(float(baseline_future_f1), 6),
        "future_macro_f1": round(float(retrained_future_f1), 6),
        "future_macro_f1_gain": round(float(retrained_future_f1 - baseline_future_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_future_pred,
            retrained_future_pred,
        ),
    }


def rolling_svc_bge_disagreement_backtest(rows, vectors: np.ndarray, schema) -> dict:
    """固定五窗口、固定20%预算验证异构 SVC-BGE disagreement 的跨期稳定性。"""
    windows = [
        rolling_svc_bge_disagreement_experiment(
            rows,
            vectors,
            schema,
            end_fraction=fraction,
        )
        for fraction in END_FRACTIONS
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    return {
        "protocol": "fixed_five_window_train_only_svc_bge_disagreement_backtest",
        "external_touched": False,
        "review_budget_fraction_predeclared": REVIEW_BUDGET_FRACTION,
        "end_fractions": list(END_FRACTIONS),
        "windows": windows,
        "summary": {
            "window_count": len(gains),
            "mean_future_macro_f1_gain": round(float(np.mean(gains)), 6),
            "median_future_macro_f1_gain": round(float(np.median(gains)), 6),
            "min_future_macro_f1_gain": round(float(np.min(gains)), 6),
            "max_future_macro_f1_gain": round(float(np.max(gains)), 6),
            "positive_windows": int(sum(gain > 0 for gain in gains)),
            "all_windows_positive": bool(all(gain > 0 for gain in gains)),
        },
    }


def rolling_svc_bge_hybrid_experiment(
    rows,
    vectors: np.ndarray,
    schema,
    *,
    end_fraction: float,
) -> dict:
    """固定20%总预算、固定50/50 disagreement + temporal coverage 的单点验证。"""
    (
        history_idx,
        review_idx,
        future_idx,
        dropped_idx,
        review_cutoff,
        future_cutoff,
        end_time,
    ) = temporal_three_way_group_split_at_end(
        rows,
        end_fraction=end_fraction,
        review_ratio=0.15,
        future_ratio=0.15,
    )
    history_texts = [production_like_text(rows[i]) for i in history_idx]
    history_labels = [str(rows[i].event_label) for i in history_idx]
    review_texts = [production_like_text(rows[i]) for i in review_idx]
    review_labels = [str(rows[i].event_label) for i in review_idx]
    future_texts = [production_like_text(rows[i]) for i in future_idx]
    future_labels = [str(rows[i].event_label) for i in future_idx]
    classes = sorted(set(history_labels))
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    primary = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    primary_review_pred = primary.predict(review_texts).tolist()
    review_scores = np.asarray(primary.decision_function(review_texts), dtype=np.float64)
    order = np.argsort(-review_scores, axis=1)
    margins = review_scores[np.arange(len(review_scores)), order[:, 0]] - review_scores[
        np.arange(len(review_scores)), order[:, 1]
    ]
    secondary_review_pred = _top1_exemplar_prediction(
        np.asarray(vectors)[history_idx],
        history_labels,
        np.asarray(vectors)[review_idx],
        classes,
    )
    budget = max(1, int(round(len(review_idx) * REVIEW_BUDGET_FRACTION)))
    selected_local, disagreement_selected_count, coverage_selected_count = (
        _hybrid_disagreement_temporal_indices(
            primary_review_pred,
            secondary_review_pred,
            margins,
            [rows[i].publish_time for i in review_idx],
            count=budget,
        )
    )
    selected_texts = [review_texts[i] for i in selected_local]
    selected_labels = [review_labels[i] for i in selected_local]
    selected_errors = sum(primary_review_pred[i] != review_labels[i] for i in selected_local)

    baseline_future_pred = primary.predict(future_texts).tolist()
    baseline_future_f1 = f1_score(future_labels, baseline_future_pred, average="macro", zero_division=0)
    retrained = fit_with_descriptions(
        history_texts + selected_texts,
        history_labels + selected_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    retrained_future_pred = retrained.predict(future_texts).tolist()
    retrained_future_f1 = f1_score(
        future_labels,
        retrained_future_pred,
        average="macro",
        zero_division=0,
    )
    disagreement_count = sum(
        left != right for left, right in zip(primary_review_pred, secondary_review_pred)
    )
    return {
        "protocol": "train_only_svc_bge_disagreement_plus_temporal_coverage_review",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "review_window_count": len(review_idx),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "review_budget_fraction": REVIEW_BUDGET_FRACTION,
        "hybrid_disagreement_share_predeclared": HYBRID_DISAGREEMENT_SHARE,
        "requested_review_count": budget,
        "disagreement_count": disagreement_count,
        "disagreement_selected_count": disagreement_selected_count,
        "coverage_selected_count": coverage_selected_count,
        "selected_count": len(selected_local),
        "selection": "half_svc_bge_disagreement_class_balance_margin_plus_half_predicted_class_temporal_coverage",
        "selection_uses_review_gold": False,
        "selection_uses_future_gold": False,
        "selected_primary_error_rate": round(selected_errors / max(1, len(selected_local)), 6),
        "baseline_future_macro_f1": round(float(baseline_future_f1), 6),
        "future_macro_f1": round(float(retrained_future_f1), 6),
        "future_macro_f1_gain": round(float(retrained_future_f1 - baseline_future_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_future_pred,
            retrained_future_pred,
        ),
    }


def rolling_svc_bge_hybrid_backtest(rows, vectors: np.ndarray, schema) -> dict:
    """固定五窗口验证 20% 总预算、50/50 异构 disagreement 与时间覆盖混合策略。"""
    windows = [
        rolling_svc_bge_hybrid_experiment(rows, vectors, schema, end_fraction=fraction)
        for fraction in END_FRACTIONS
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    return {
        "protocol": "fixed_five_window_train_only_svc_bge_hybrid_backtest",
        "external_touched": False,
        "review_budget_fraction_predeclared": REVIEW_BUDGET_FRACTION,
        "hybrid_disagreement_share_predeclared": HYBRID_DISAGREEMENT_SHARE,
        "end_fractions": list(END_FRACTIONS),
        "windows": windows,
        "summary": {
            "window_count": len(gains),
            "mean_future_macro_f1_gain": round(float(np.mean(gains)), 6),
            "median_future_macro_f1_gain": round(float(np.median(gains)), 6),
            "min_future_macro_f1_gain": round(float(np.min(gains)), 6),
            "max_future_macro_f1_gain": round(float(np.max(gains)), 6),
            "positive_windows": int(sum(gain > 0 for gain in gains)),
            "all_windows_positive": bool(all(gain > 0 for gain in gains)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    rows = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    manifest, vectors = load_exported_vectors(args.train_embeddings_dir)
    if manifest.article_count != len(rows):
        raise ValueError("train embedding count mismatch")
    if load_exported_article_ids(args.train_embeddings_dir) != [row.article_id for row in rows]:
        raise ValueError("train embedding order mismatch")
    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    if args.hybrid:
        result = rolling_svc_bge_hybrid_backtest(rows, np.asarray(vectors), schema)
    else:
        result = rolling_svc_bge_disagreement_backtest(rows, np.asarray(vectors), schema)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
