from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from tools.benchmark_schema_constrained_svc import fit_with_descriptions, label_description_examples
from eventlens.config import load_settings
from eventlens.event_retrieval import EventSchemaIndex
from eventlens.io import read_competition_labeled_excel
from eventlens.preprocess import clean_text


def duplication_groups(rows) -> list[str]:
    return [
        str(row.duplication_id).strip()
        if row.duplication_id not in (None, "")
        else f"article:{row.article_id}"
        for row in rows
    ]


def production_like_text(row, *, max_content_chars: int = 2400) -> str:
    parts = [row.title, row.source, row.content[:max_content_chars]]
    return " ".join(clean_text(part) for part in parts if clean_text(part))


def temporal_group_split(rows, *, holdout_ratio: float) -> tuple[list[int], list[int], list[int], str]:
    """严格按时间切分；跨越 cutoff 的 duplication group 直接丢弃。"""
    if not 0.05 <= holdout_ratio <= 0.5:
        raise ValueError("holdout_ratio must be in [0.05, 0.5]")
    groups = duplication_groups(rows)
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_indices[group].append(index)
    group_ranges = []
    for group, indices in group_indices.items():
        times = [rows[i].publish_time for i in indices if rows[i].publish_time is not None]
        if not times:
            raise ValueError(f"group {group} has no publish_time")
        group_ranges.append((min(times), max(times), group))
    all_times = sorted(row.publish_time for row in rows if row.publish_time is not None)
    if len(all_times) != len(rows):
        raise ValueError("some rows have no publish_time")
    cutoff_index = min(len(all_times) - 1, int(len(all_times) * (1.0 - holdout_ratio)))
    cutoff_time = all_times[cutoff_index]
    train_groups = {group for start, end, group in group_ranges if end < cutoff_time}
    val_groups = {group for start, end, group in group_ranges if start >= cutoff_time}
    dropped_groups = {group for start, end, group in group_ranges if start < cutoff_time <= end}
    if not train_groups or not val_groups:
        raise ValueError("temporal split left an empty side")
    train_idx = [i for i, group in enumerate(groups) if group in train_groups]
    val_idx = [i for i, group in enumerate(groups) if group in val_groups]
    dropped_idx = [i for i, group in enumerate(groups) if group in dropped_groups]
    return train_idx, val_idx, dropped_idx, cutoff_time.isoformat()


def temporal_three_way_group_split(
    rows,
    *,
    review_ratio: float = 0.15,
    future_ratio: float = 0.15,
) -> tuple[list[int], list[int], list[int], list[int], str, str]:
    """历史/复核/未来三段严格时间切分；跨任一边界的 duplication group 丢弃。"""
    if not 0.05 <= review_ratio <= 0.3 or not 0.05 <= future_ratio <= 0.3:
        raise ValueError("review_ratio and future_ratio must be in [0.05, 0.3]")
    if review_ratio + future_ratio >= 0.5:
        raise ValueError("review_ratio + future_ratio must be < 0.5")
    groups = duplication_groups(rows)
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_indices[group].append(index)
    all_times = sorted(row.publish_time for row in rows if row.publish_time is not None)
    if len(all_times) != len(rows):
        raise ValueError("some rows have no publish_time")
    first_idx = min(len(all_times) - 2, int(len(all_times) * (1.0 - review_ratio - future_ratio)))
    second_idx = min(len(all_times) - 1, int(len(all_times) * (1.0 - future_ratio)))
    first_cutoff = all_times[first_idx]
    second_cutoff = all_times[second_idx]
    if first_cutoff >= second_cutoff:
        raise ValueError("temporal cutoffs collapsed")

    history_groups: set[str] = set()
    review_groups: set[str] = set()
    future_groups: set[str] = set()
    dropped_groups: set[str] = set()
    for group, indices in group_indices.items():
        times = [rows[i].publish_time for i in indices if rows[i].publish_time is not None]
        if not times:
            raise ValueError(f"group {group} has no publish_time")
        start, end = min(times), max(times)
        if end < first_cutoff:
            history_groups.add(group)
        elif start >= first_cutoff and end < second_cutoff:
            review_groups.add(group)
        elif start >= second_cutoff:
            future_groups.add(group)
        else:
            dropped_groups.add(group)
    if not history_groups or not review_groups or not future_groups:
        raise ValueError("three-way temporal split left an empty side")
    history_idx = [i for i, group in enumerate(groups) if group in history_groups]
    review_idx = [i for i, group in enumerate(groups) if group in review_groups]
    future_idx = [i for i, group in enumerate(groups) if group in future_groups]
    dropped_idx = [i for i, group in enumerate(groups) if group in dropped_groups]
    return (
        history_idx,
        review_idx,
        future_idx,
        dropped_idx,
        first_cutoff.isoformat(),
        second_cutoff.isoformat(),
    )


def temporal_three_way_group_split_at_end(
    rows,
    *,
    end_fraction: float,
    review_ratio: float = 0.15,
    future_ratio: float = 0.15,
) -> tuple[list[int], list[int], list[int], list[int], str, str, str]:
    """固定结束分位的三段时间回测；结束边界之后的样本完全不参与该窗口。"""
    if not 0.6 <= end_fraction <= 1.0:
        raise ValueError("end_fraction must be in [0.6, 1.0]")
    if review_ratio + future_ratio >= end_fraction:
        raise ValueError("review_ratio + future_ratio must be smaller than end_fraction")
    all_times = sorted(row.publish_time for row in rows if row.publish_time is not None)
    if len(all_times) != len(rows):
        raise ValueError("some rows have no publish_time")
    end_index = min(len(all_times) - 1, max(1, int(len(all_times) * end_fraction) - 1))
    end_time = all_times[end_index]
    first_fraction = end_fraction - review_ratio - future_ratio
    second_fraction = end_fraction - future_ratio
    first_index = min(len(all_times) - 2, max(0, int(len(all_times) * first_fraction)))
    second_index = min(len(all_times) - 1, max(first_index + 1, int(len(all_times) * second_fraction)))
    first_cutoff = all_times[first_index]
    second_cutoff = all_times[second_index]
    if not first_cutoff < second_cutoff <= end_time:
        raise ValueError("rolling temporal cutoffs collapsed")

    groups = duplication_groups(rows)
    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_indices[group].append(index)
    history_groups: set[str] = set()
    review_groups: set[str] = set()
    future_groups: set[str] = set()
    dropped_groups: set[str] = set()
    for group, indices in group_indices.items():
        times = [rows[i].publish_time for i in indices if rows[i].publish_time is not None]
        start, end = min(times), max(times)
        if start > end_time:
            continue
        if end < first_cutoff:
            history_groups.add(group)
        elif start >= first_cutoff and end < second_cutoff:
            review_groups.add(group)
        elif start >= second_cutoff and end <= end_time:
            future_groups.add(group)
        else:
            dropped_groups.add(group)
    if not history_groups or not review_groups or not future_groups:
        raise ValueError("rolling three-way temporal split left an empty side")
    history_idx = [i for i, group in enumerate(groups) if group in history_groups]
    review_idx = [i for i, group in enumerate(groups) if group in review_groups]
    future_idx = [i for i, group in enumerate(groups) if group in future_groups]
    dropped_idx = [i for i, group in enumerate(groups) if group in dropped_groups]
    return (
        history_idx,
        review_idx,
        future_idx,
        dropped_idx,
        first_cutoff.isoformat(),
        second_cutoff.isoformat(),
        end_time.isoformat(),
    )


def recent_repeat_indices(rows, train_idx: list[int], *, recent_ratio: float = 0.25) -> list[int]:
    """固定一次重采样最近训练样本；仅依赖训练时间，不读取验证标签。"""
    if not 0.05 <= recent_ratio <= 0.5:
        raise ValueError("recent_ratio must be in [0.05, 0.5]")
    ordered = sorted(train_idx, key=lambda i: rows[i].publish_time)
    repeat_count = max(1, int(round(len(ordered) * recent_ratio)))
    return ordered[-repeat_count:]


def recent_group_safe_history_indices(
    rows,
    history_idx: list[int],
    *,
    keep_fraction: float = 0.75,
) -> tuple[list[int], list[int], str]:
    """只保留最近一段 history，跨 cutoff 的 duplication group 丢弃，避免时间泄漏。"""
    if not 0.5 <= keep_fraction < 1.0:
        raise ValueError("keep_fraction must be in [0.5, 1.0)")
    if not history_idx:
        raise ValueError("history_idx must not be empty")
    groups = duplication_groups(rows)
    ordered_times = sorted(rows[i].publish_time for i in history_idx)
    if any(value is None for value in ordered_times):
        raise ValueError("some history rows have no publish_time")
    cutoff_pos = min(len(ordered_times) - 1, int(len(ordered_times) * (1.0 - keep_fraction)))
    cutoff = ordered_times[cutoff_pos]

    by_group: dict[str, list[int]] = defaultdict(list)
    for index in history_idx:
        by_group[groups[index]].append(index)
    kept_groups: set[str] = set()
    dropped_groups: set[str] = set()
    for group, indices in by_group.items():
        times = [rows[i].publish_time for i in indices]
        if min(times) >= cutoff:
            kept_groups.add(group)
        elif min(times) < cutoff <= max(times):
            dropped_groups.add(group)
    kept = [i for i in history_idx if groups[i] in kept_groups]
    dropped = [i for i in history_idx if groups[i] in dropped_groups]
    if not kept:
        raise ValueError("recent history selection left no rows")
    return kept, dropped, cutoff.isoformat()


def temporal_error_diagnostics(val_labels: list[str], pred: list[str]) -> dict:
    labels = sorted(set(val_labels) | set(pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        val_labels,
        pred,
        labels=labels,
        zero_division=0,
    )
    per_class = [
        {
            "label": label,
            "support": int(class_support),
            "precision": round(float(class_precision), 6),
            "recall": round(float(class_recall), 6),
            "f1": round(float(class_f1), 6),
        }
        for label, class_precision, class_recall, class_f1, class_support in zip(
            labels, precision, recall, f1, support
        )
    ]
    matrix = confusion_matrix(val_labels, pred, labels=labels)
    confusion_pairs = []
    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            if true_index == pred_index:
                continue
            count = int(matrix[true_index, pred_index])
            if count:
                confusion_pairs.append(
                    {"true_label": true_label, "pred_label": pred_label, "count": count}
                )
    confusion_pairs.sort(key=lambda item: (-item["count"], item["true_label"], item["pred_label"]))
    return {
        "worst_supported_classes": sorted(
            (item for item in per_class if item["support"] >= 3),
            key=lambda item: (item["f1"], -item["support"], item["label"]),
        )[:10],
        "top_confusion_pairs": confusion_pairs[:15],
    }


def _balanced_low_margin_indices(pred: list[str], margins: np.ndarray, count: int) -> list[int]:
    """按预测类轮转抽取低 margin 样本；选样过程不读取 Gold。"""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(pred):
        groups[label].append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: (float(margins[i]), i))
    selected: list[int] = []
    offset = 0
    ordered_labels = sorted(groups)
    while len(selected) < count:
        added = False
        for label in ordered_labels:
            indices = groups[label]
            if offset < len(indices):
                selected.append(indices[offset])
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
        offset += 1
    return selected


def _disagreement_class_balanced_indices(
    primary_pred: list[str],
    secondary_pred: list[str],
    margins: np.ndarray,
    count: int,
) -> list[int]:
    """仅在两模型分歧样本中按主模型预测类轮转；不读取 Gold。"""
    if len(primary_pred) != len(secondary_pred) or len(primary_pred) != len(margins):
        raise ValueError("predictions and margins must be aligned")
    disagreement = [i for i, (left, right) in enumerate(zip(primary_pred, secondary_pred)) if left != right]
    groups: dict[str, list[int]] = defaultdict(list)
    for index in disagreement:
        groups[primary_pred[index]].append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: (float(margins[i]), i))
    selected: list[int] = []
    offset = 0
    ordered_labels = sorted(groups)
    while len(selected) < count:
        added = False
        for label in ordered_labels:
            indices = groups[label]
            if offset < len(indices):
                selected.append(indices[offset])
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
        offset += 1
    return selected


def _history_group_fold_indices(rows, history_idx: list[int], *, n_folds: int = 3) -> list[list[int]]:
    """仅用 history duplication group 做稳定哈希分桶，构造 chronology-safe 子模型。"""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    groups = duplication_groups(rows)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for index in history_idx:
        digest = hashlib.sha256(groups[index].encode("utf-8")).digest()
        fold = int.from_bytes(digest[:4], "big") % n_folds
        folds[fold].append(index)
    if any(not fold for fold in folds):
        raise ValueError("history group folds must all be non-empty")
    return folds


def _majority_vote(predictions: list[list[str]]) -> list[str]:
    """确定性多数投票；平票时按标签字典序，避免隐式随机性。"""
    if not predictions or not predictions[0]:
        raise ValueError("predictions must be non-empty")
    width = len(predictions[0])
    if any(len(items) != width for items in predictions):
        raise ValueError("predictions must be aligned")
    voted: list[str] = []
    for column in zip(*predictions):
        counts: dict[str, int] = defaultdict(int)
        for label in column:
            counts[label] += 1
        voted.append(sorted(counts, key=lambda label: (-counts[label], label))[0])
    return voted


def _risk_weighted_low_margin_indices(
    pred: list[str],
    margins: np.ndarray,
    class_error_rate: dict[str, float],
    count: int,
) -> list[int]:
    """按 history OOF 预测类错误率分配预算，再在类内取低 margin；不读取 review Gold。"""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(pred):
        groups[label].append(index)
    for indices in groups.values():
        indices.sort(key=lambda i: (float(margins[i]), i))
    weights = {
        label: max(0.0, float(class_error_rate.get(label, 0.0))) * len(indices)
        for label, indices in groups.items()
    }
    if sum(weights.values()) <= 0.0:
        return _balanced_low_margin_indices(pred, margins, count)

    selected: list[int] = []
    remaining = {label: list(indices) for label, indices in groups.items()}
    while len(selected) < count:
        candidates = [label for label, indices in remaining.items() if indices]
        if not candidates:
            break
        # 每次选择“当前应得预算 - 已选数量”最大的类，避免一次性 rounding 丢预算。
        total_weight = sum(weights[label] for label in candidates)
        if total_weight <= 0.0:
            label = sorted(candidates)[0]
        else:
            target_total = len(selected) + 1
            label = sorted(
                candidates,
                key=lambda item: (
                    -(target_total * weights[item] / total_weight - (len(groups[item]) - len(remaining[item]))),
                    item,
                ),
            )[0]
        selected.append(remaining[label].pop(0))
    return selected


def _predicted_class_balanced_temporal_coverage_indices(
    predicted_labels: list[str],
    publish_times,
    *,
    count: int,
) -> list[int]:
    """按预测类均衡配额，并在每类时间轴上均匀取点；不读取 Gold 或 margin。"""
    if len(predicted_labels) != len(publish_times):
        raise ValueError("predicted_labels and publish_times must be aligned")
    if count <= 0 or not predicted_labels:
        return []
    count = min(count, len(predicted_labels))
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(predicted_labels):
        by_class[str(label)].append(index)
    for label in by_class:
        by_class[label].sort(key=lambda i: (publish_times[i], i))

    labels = sorted(by_class)
    quotas = {label: min(len(by_class[label]), count // len(labels)) for label in labels}
    remaining = count - sum(quotas.values())
    while remaining > 0:
        progressed = False
        for label in labels:
            if quotas[label] >= len(by_class[label]):
                continue
            quotas[label] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break

    selected: list[int] = []
    for label in labels:
        candidates = by_class[label]
        quota = quotas[label]
        if quota == 0:
            continue
        positions = (
            [len(candidates) // 2]
            if quota == 1
            else np.linspace(0, len(candidates) - 1, num=quota).round().astype(int).tolist()
        )
        selected.extend(candidates[position] for position in positions)
    return sorted(selected, key=lambda i: (publish_times[i], i))


def _history_oof_predicted_class_error_rate(rows, history_idx: list[int], schema) -> dict[str, float]:
    """仅用 history 的 group-safe 3-fold OOF 估计预测类错误率。"""
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    folds = _history_group_fold_indices(rows, history_idx, n_folds=3)
    history_set = set(history_idx)
    totals: dict[str, int] = defaultdict(int)
    errors: dict[str, int] = defaultdict(int)
    for heldout_fold in folds:
        train_idx = sorted(history_set - set(heldout_fold))
        model = fit_with_descriptions(
            [production_like_text(rows[i]) for i in train_idx],
            [str(rows[i].event_label) for i in train_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        pred = model.predict([production_like_text(rows[i]) for i in heldout_fold]).tolist()
        for index, label in zip(heldout_fold, pred):
            totals[label] += 1
            errors[label] += int(label != str(rows[index].event_label))
    return {
        label: errors[label] / totals[label]
        for label in totals
        if totals[label] > 0
    }


def review_oracle_frontier(val_labels: list[str], pred: list[str], scores: np.ndarray) -> list[dict]:
    """评估人工复核理论收益；Gold 仅用于 oracle 覆盖后的指标计算。"""
    order = np.argsort(-scores, axis=1)
    margins = scores[np.arange(len(scores)), order[:, 0]] - scores[np.arange(len(scores)), order[:, 1]]
    baseline = f1_score(val_labels, pred, average="macro", zero_division=0)
    frontier = []
    for fraction in (0.15, 0.20):
        count = max(1, int(round(len(pred) * fraction)))
        selected = _balanced_low_margin_indices(pred, margins, count)
        oracle_pred = list(pred)
        errors = 0
        for index in selected:
            if oracle_pred[index] != val_labels[index]:
                errors += 1
            oracle_pred[index] = val_labels[index]
        oracle = f1_score(val_labels, oracle_pred, average="macro", zero_division=0)
        frontier.append(
            {
                "fraction": fraction,
                "selected_count": len(selected),
                "selection": "predicted_class_balanced_low_margin",
                "selection_uses_gold": False,
                "selected_error_rate": round(errors / max(1, len(selected)), 6),
                "oracle_macro_f1": round(float(oracle), 6),
                "oracle_macro_f1_gain": round(float(oracle - baseline), 6),
            }
        )
    return frontier


def paired_bootstrap_macro_f1_gain(
    truth: list[str],
    baseline_pred: list[str],
    candidate_pred: list[str],
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260829,
) -> dict:
    """对同一未来窗口做配对 bootstrap；不改变模型选择，只量化增益不确定性。"""
    if not (len(truth) == len(baseline_pred) == len(candidate_pred)) or not truth:
        raise ValueError("truth and predictions must be non-empty and aligned")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")
    rng = np.random.default_rng(seed)
    truth_arr = np.asarray(truth, dtype=object)
    base_arr = np.asarray(baseline_pred, dtype=object)
    cand_arr = np.asarray(candidate_pred, dtype=object)
    gains = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(truth_arr), size=len(truth_arr))
        sampled_truth = truth_arr[sampled]
        base_f1 = f1_score(sampled_truth, base_arr[sampled], average="macro", zero_division=0)
        cand_f1 = f1_score(sampled_truth, cand_arr[sampled], average="macro", zero_division=0)
        gains[index] = cand_f1 - base_f1
    point_gain = f1_score(truth, candidate_pred, average="macro", zero_division=0) - f1_score(
        truth, baseline_pred, average="macro", zero_division=0
    )
    lower, upper = np.quantile(gains, [0.025, 0.975])
    return {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "point_gain": round(float(point_gain), 6),
        "gain_ci95": [round(float(lower), 6), round(float(upper), 6)],
        "positive_gain_probability": round(float(np.mean(gains > 0.0)), 6),
    }


def evaluate_model(train_texts, train_labels, val_texts, val_labels, desc_texts, desc_labels) -> dict:
    model = fit_with_descriptions(train_texts, train_labels, desc_texts, desc_labels, repeat=1)
    pred = model.predict(val_texts).tolist()
    scores = np.asarray(model.decision_function(val_texts), dtype=np.float64)
    return {
        "accuracy": round(accuracy_score(val_labels, pred), 6),
        "macro_f1": round(f1_score(val_labels, pred, average="macro", zero_division=0), 6),
        "diagnostics": temporal_error_diagnostics(val_labels, pred),
        "review_oracle_frontier": review_oracle_frontier(val_labels, pred, scores),
    }


def rolling_review_tranche_experiment(rows, schema, *, end_fraction: float = 1.0) -> dict:
    """用中间时间窗模拟 approved Gold，验证其对更未来窗口的真实增益。"""
    if end_fraction == 1.0:
        history_idx, review_idx, future_idx, dropped_idx, review_cutoff, future_cutoff = (
            temporal_three_way_group_split(rows, review_ratio=0.15, future_ratio=0.15)
        )
        end_time = max(row.publish_time for row in rows).isoformat()
    else:
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
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    base_model = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    review_pred = base_model.predict(review_texts).tolist()
    review_scores = np.asarray(base_model.decision_function(review_texts), dtype=np.float64)
    order = np.argsort(-review_scores, axis=1)
    margins = review_scores[np.arange(len(review_scores)), order[:, 0]] - review_scores[
        np.arange(len(review_scores)), order[:, 1]
    ]
    future_pred = base_model.predict(future_texts).tolist()
    baseline_future_f1 = f1_score(future_labels, future_pred, average="macro", zero_division=0)

    tranches = []
    for fraction in (0.15, 0.20):
        count = max(1, int(round(len(review_idx) * fraction)))
        selected_local = _balanced_low_margin_indices(review_pred, margins, count)
        selected_texts = [review_texts[i] for i in selected_local]
        selected_labels = [review_labels[i] for i in selected_local]
        selected_errors = sum(review_pred[i] != review_labels[i] for i in selected_local)
        retrained = fit_with_descriptions(
            history_texts + selected_texts,
            history_labels + selected_labels,
            desc_texts,
            desc_labels,
            repeat=1,
        )
        retrained_pred = retrained.predict(future_texts).tolist()
        retrained_f1 = f1_score(future_labels, retrained_pred, average="macro", zero_division=0)
        tranches.append(
            {
                "fraction_of_review_window": fraction,
                "selected_count": len(selected_local),
                "selection": "predicted_class_balanced_low_margin",
                "selection_uses_gold": False,
                "selected_error_rate": round(selected_errors / max(1, len(selected_local)), 6),
                "future_macro_f1": round(float(retrained_f1), 6),
                "future_macro_f1_gain": round(float(retrained_f1 - baseline_future_f1), 6),
                "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
                    future_labels,
                    future_pred,
                    retrained_pred,
                ),
            }
        )
    groups = duplication_groups(rows)
    return {
        "protocol": "train_only_rolling_reviewed_gold_three_way_temporal",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "review_window_count": len(review_idx),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "group_overlap": len(
            ({groups[i] for i in history_idx} & {groups[i] for i in review_idx})
            | ({groups[i] for i in history_idx} & {groups[i] for i in future_idx})
            | ({groups[i] for i in review_idx} & {groups[i] for i in future_idx})
        ),
        "baseline_future_macro_f1": round(float(baseline_future_f1), 6),
        "tranches": tranches,
    }


def rolling_review_backtest(rows, schema) -> dict:
    """固定三个结束分位复验 reviewed-Gold 收益，避免单窗口偶然性。"""
    windows = [rolling_review_tranche_experiment(rows, schema, end_fraction=fraction) for fraction in (0.70, 0.85, 1.0)]
    summary = []
    for fraction in (0.15, 0.20):
        gains = [
            next(item for item in window["tranches"] if item["fraction_of_review_window"] == fraction)[
                "future_macro_f1_gain"
            ]
            for window in windows
        ]
        summary.append(
            {
                "fraction_of_review_window": fraction,
                "window_count": len(gains),
                "mean_future_macro_f1_gain": round(float(np.mean(gains)), 6),
                "min_future_macro_f1_gain": round(float(np.min(gains)), 6),
                "positive_windows": int(sum(gain > 0 for gain in gains)),
                "all_windows_positive": bool(all(gain > 0 for gain in gains)),
            }
        )
    return {
        "protocol": "fixed_three_window_train_only_rolling_review_backtest",
        "external_touched": False,
        "end_fractions": [0.70, 0.85, 1.0],
        "windows": windows,
        "summary": summary,
    }


def rolling_review_robustness_backtest(rows, schema) -> dict:
    """固定五个结束分位，仅复验既定 20% review tranche 的跨期稳健性，不做预算搜索。"""
    end_fractions = (0.65, 0.75, 0.85, 0.95, 1.0)
    windows = [
        rolling_review_tranche_experiment(rows, schema, end_fraction=fraction)
        for fraction in end_fractions
    ]
    selected = [
        next(
            item
            for item in window["tranches"]
            if item["fraction_of_review_window"] == 0.20
        )
        for window in windows
    ]
    gains = [item["future_macro_f1_gain"] for item in selected]
    return {
        "protocol": "fixed_five_window_train_only_review_20pct_robustness_backtest",
        "external_touched": False,
        "selection_budget_predeclared": 0.20,
        "end_fractions": list(end_fractions),
        "windows": [
            {
                "end_fraction": window["end_fraction"],
                "window_end_publish_time": window["window_end_publish_time"],
                "history_count": window["history_count"],
                "review_window_count": window["review_window_count"],
                "future_count": window["future_count"],
                "group_overlap": window["group_overlap"],
                "baseline_future_macro_f1": window["baseline_future_macro_f1"],
                "review_20pct": tranche,
            }
            for window, tranche in zip(windows, selected)
        ],
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


def rolling_full_review_refresh_experiment(rows, schema, *, end_fraction: float) -> dict:
    """中间时间窗 100% 人工审批为 Gold 后回流；仅量化近期 Gold refresh 上限，不做选样。"""
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
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    baseline = fit_with_descriptions(
        history_texts,
        history_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    baseline_future_pred = baseline.predict(future_texts).tolist()
    baseline_future_f1 = f1_score(
        future_labels,
        baseline_future_pred,
        average="macro",
        zero_division=0,
    )
    refreshed = fit_with_descriptions(
        history_texts + review_texts,
        history_labels + review_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    refreshed_future_pred = refreshed.predict(future_texts).tolist()
    refreshed_future_f1 = f1_score(
        future_labels,
        refreshed_future_pred,
        average="macro",
        zero_division=0,
    )
    groups = duplication_groups(rows)
    return {
        "protocol": "train_only_full_review_window_gold_refresh_upper_bound",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "approved_gold_refresh_count": len(review_idx),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "group_overlap": len(
            ({groups[i] for i in history_idx} & {groups[i] for i in review_idx})
            | ({groups[i] for i in history_idx} & {groups[i] for i in future_idx})
            | ({groups[i] for i in review_idx} & {groups[i] for i in future_idx})
        ),
        "review_fraction_used": 1.0,
        "selection": "all_review_window_rows_assumed_human_approved_gold",
        "selection_uses_future_gold": False,
        "baseline_future_macro_f1": round(float(baseline_future_f1), 6),
        "future_macro_f1": round(float(refreshed_future_f1), 6),
        "future_macro_f1_gain": round(float(refreshed_future_f1 - baseline_future_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_future_pred,
            refreshed_future_pred,
        ),
    }


def rolling_full_review_refresh_backtest(rows, schema) -> dict:
    """固定五窗口验证 100% approved recent Gold refresh 是否跨期稳定，作为数据飞轮上限诊断。"""
    end_fractions = (0.65, 0.75, 0.85, 0.95, 1.0)
    windows = [
        rolling_full_review_refresh_experiment(rows, schema, end_fraction=fraction)
        for fraction in end_fractions
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    return {
        "protocol": "fixed_five_window_train_only_full_review_gold_refresh_backtest",
        "external_touched": False,
        "review_fraction_predeclared": 1.0,
        "end_fractions": list(end_fractions),
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


def rolling_temporal_coverage_review_experiment(
    rows,
    schema,
    *,
    end_fraction: float,
    review_fraction: float = 0.50,
) -> dict:
    """固定预算 review：仅按预测类均衡 + 时间覆盖选人工 Gold，验证低成本 refresh。"""
    if not 0 < review_fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")
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
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    baseline = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    baseline_future_pred = baseline.predict(future_texts).tolist()
    baseline_future_f1 = f1_score(future_labels, baseline_future_pred, average="macro", zero_division=0)
    review_pred = baseline.predict(review_texts).tolist()
    budget = max(1, int(round(len(review_idx) * review_fraction)))
    selected_local = _predicted_class_balanced_temporal_coverage_indices(
        review_pred,
        [rows[i].publish_time for i in review_idx],
        count=budget,
    )
    selected_texts = [review_texts[i] for i in selected_local]
    selected_labels = [review_labels[i] for i in selected_local]
    selected_errors = sum(review_pred[i] != review_labels[i] for i in selected_local)
    refreshed = fit_with_descriptions(
        history_texts + selected_texts,
        history_labels + selected_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    refreshed_future_pred = refreshed.predict(future_texts).tolist()
    refreshed_future_f1 = f1_score(
        future_labels,
        refreshed_future_pred,
        average="macro",
        zero_division=0,
    )
    groups = duplication_groups(rows)
    return {
        "protocol": "train_only_predicted_class_balanced_temporal_coverage_review",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "review_window_count": len(review_idx),
        "approved_gold_refresh_count": len(selected_local),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "group_overlap": len(
            ({groups[i] for i in history_idx} & {groups[i] for i in review_idx})
            | ({groups[i] for i in history_idx} & {groups[i] for i in future_idx})
            | ({groups[i] for i in review_idx} & {groups[i] for i in future_idx})
        ),
        "review_fraction_predeclared": review_fraction,
        "selection": "predicted_class_balance_plus_within_class_temporal_coverage",
        "selection_uses_review_gold": False,
        "selection_uses_future_gold": False,
        "selected_primary_error_rate": round(selected_errors / len(selected_local), 6),
        "baseline_future_macro_f1": round(float(baseline_future_f1), 6),
        "future_macro_f1": round(float(refreshed_future_f1), 6),
        "future_macro_f1_gain": round(float(refreshed_future_f1 - baseline_future_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_future_pred,
            refreshed_future_pred,
        ),
    }


def rolling_temporal_coverage_review_backtest(
    rows,
    schema,
    *,
    review_fraction: float = 0.50,
) -> dict:
    """固定五窗口、固定 review 预算，验证时间覆盖型 approved Gold refresh 的跨期稳定性。"""
    if not 0 < review_fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")
    end_fractions = (0.65, 0.75, 0.85, 0.95, 1.0)
    windows = [
        rolling_temporal_coverage_review_experiment(
            rows,
            schema,
            end_fraction=fraction,
            review_fraction=review_fraction,
        )
        for fraction in end_fractions
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    return {
        "protocol": "fixed_five_window_train_only_temporal_coverage_review_backtest",
        "external_touched": False,
        "review_fraction_predeclared": review_fraction,
        "end_fractions": list(end_fractions),
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


def rolling_snapshot_disagreement_experiment(rows, schema, *, end_fraction: float) -> dict:
    """全历史与近期历史模型分歧驱动复核；固定 20% review budget，不扫参。"""
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
    history_idx = sorted(history_idx, key=lambda i: rows[i].publish_time)
    recent_start = len(history_idx) // 2
    recent_idx = history_idx[recent_start:]
    history_texts = [production_like_text(rows[i]) for i in history_idx]
    history_labels = [str(rows[i].event_label) for i in history_idx]
    recent_texts = [production_like_text(rows[i]) for i in recent_idx]
    recent_labels = [str(rows[i].event_label) for i in recent_idx]
    review_texts = [production_like_text(rows[i]) for i in review_idx]
    review_labels = [str(rows[i].event_label) for i in review_idx]
    future_texts = [production_like_text(rows[i]) for i in future_idx]
    future_labels = [str(rows[i].event_label) for i in future_idx]
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    primary = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    secondary = fit_with_descriptions(recent_texts, recent_labels, desc_texts, desc_labels, repeat=1)
    primary_review_pred = primary.predict(review_texts).tolist()
    secondary_review_pred = secondary.predict(review_texts).tolist()
    review_scores = np.asarray(primary.decision_function(review_texts), dtype=np.float64)
    order = np.argsort(-review_scores, axis=1)
    margins = review_scores[np.arange(len(review_scores)), order[:, 0]] - review_scores[
        np.arange(len(review_scores)), order[:, 1]
    ]
    budget = max(1, int(round(len(review_idx) * 0.20)))
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
    disagreement_count = sum(left != right for left, right in zip(primary_review_pred, secondary_review_pred))
    return {
        "protocol": "train_only_temporal_snapshot_disagreement_class_balanced_review",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "recent_snapshot_count": len(recent_idx),
        "recent_snapshot_fraction": 0.5,
        "review_window_count": len(review_idx),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "review_budget_fraction": 0.20,
        "requested_review_count": budget,
        "disagreement_count": disagreement_count,
        "selected_count": len(selected_local),
        "selection": "full_history_vs_recent_half_disagreement_then_primary_class_balance_and_margin",
        "selection_uses_gold": False,
        "selected_error_rate": round(selected_errors / max(1, len(selected_local)), 6),
        "baseline_future_macro_f1": round(float(baseline_future_f1), 6),
        "future_macro_f1": round(float(retrained_future_f1), 6),
        "future_macro_f1_gain": round(float(retrained_future_f1 - baseline_future_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_future_pred,
            retrained_future_pred,
        ),
    }


def rolling_snapshot_disagreement_backtest(rows, schema) -> dict:
    """固定三窗口检验 temporal snapshot disagreement 是否稳定改善未来窗。"""
    windows = [
        rolling_snapshot_disagreement_experiment(rows, schema, end_fraction=fraction)
        for fraction in (0.70, 0.85, 1.0)
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    return {
        "protocol": "fixed_three_window_train_only_temporal_snapshot_disagreement_backtest",
        "external_touched": False,
        "end_fractions": [0.70, 0.85, 1.0],
        "windows": windows,
        "summary": {
            "window_count": len(gains),
            "mean_future_macro_f1_gain": round(float(np.mean(gains)), 6),
            "min_future_macro_f1_gain": round(float(np.min(gains)), 6),
            "positive_windows": int(sum(gain > 0 for gain in gains)),
            "all_windows_positive": bool(all(gain > 0 for gain in gains)),
        },
    }


def rolling_history_ensemble_disagreement_experiment(rows, schema, *, end_fraction: float) -> dict:
    """仅 history 内 3-fold 子模型集成分歧驱动 review，避免 snapshot 偏置与未来泄漏。"""
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
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    primary = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    primary_review_pred = primary.predict(review_texts).tolist()
    review_scores = np.asarray(primary.decision_function(review_texts), dtype=np.float64)
    order = np.argsort(-review_scores, axis=1)
    margins = review_scores[np.arange(len(review_scores)), order[:, 0]] - review_scores[
        np.arange(len(review_scores)), order[:, 1]
    ]

    history_folds = _history_group_fold_indices(rows, history_idx, n_folds=3)
    member_predictions: list[list[str]] = []
    history_set = set(history_idx)
    for heldout_fold in history_folds:
        member_idx = sorted(history_set - set(heldout_fold))
        member = fit_with_descriptions(
            [production_like_text(rows[i]) for i in member_idx],
            [str(rows[i].event_label) for i in member_idx],
            desc_texts,
            desc_labels,
            repeat=1,
        )
        member_predictions.append(member.predict(review_texts).tolist())
    ensemble_review_pred = _majority_vote(member_predictions)

    budget = max(1, int(round(len(review_idx) * 0.20)))
    selected_local = _disagreement_class_balanced_indices(
        primary_review_pred,
        ensemble_review_pred,
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
        left != right for left, right in zip(primary_review_pred, ensemble_review_pred)
    )
    return {
        "protocol": "train_only_history_3fold_ensemble_disagreement_class_balanced_review",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "history_fold_counts": [len(fold) for fold in history_folds],
        "review_window_count": len(review_idx),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "review_budget_fraction": 0.20,
        "requested_review_count": budget,
        "disagreement_count": disagreement_count,
        "selected_count": len(selected_local),
        "selection": "full_history_vs_history_only_3fold_majority_disagreement_then_class_balance_and_margin",
        "selection_uses_gold": False,
        "selected_error_rate": round(selected_errors / max(1, len(selected_local)), 6),
        "baseline_future_macro_f1": round(float(baseline_future_f1), 6),
        "future_macro_f1": round(float(retrained_future_f1), 6),
        "future_macro_f1_gain": round(float(retrained_future_f1 - baseline_future_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_future_pred,
            retrained_future_pred,
        ),
    }


def rolling_history_ensemble_disagreement_backtest(rows, schema) -> dict:
    """固定三窗口验证 history-only ensemble disagreement 是否稳定改善未来窗。"""
    windows = [
        rolling_history_ensemble_disagreement_experiment(rows, schema, end_fraction=fraction)
        for fraction in (0.70, 0.85, 1.0)
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    return {
        "protocol": "fixed_three_window_train_only_history_ensemble_disagreement_backtest",
        "external_touched": False,
        "end_fractions": [0.70, 0.85, 1.0],
        "windows": windows,
        "summary": {
            "window_count": len(gains),
            "mean_future_macro_f1_gain": round(float(np.mean(gains)), 6),
            "min_future_macro_f1_gain": round(float(np.min(gains)), 6),
            "positive_windows": int(sum(gain > 0 for gain in gains)),
            "all_windows_positive": bool(all(gain > 0 for gain in gains)),
        },
    }


def rolling_history_oof_risk_experiment(rows, schema, *, end_fraction: float) -> dict:
    """固定 20% review：history OOF 预测类错误风险分配预算，类内按低 margin 选样。"""
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
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    primary = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    review_pred = primary.predict(review_texts).tolist()
    review_scores = np.asarray(primary.decision_function(review_texts), dtype=np.float64)
    order = np.argsort(-review_scores, axis=1)
    margins = review_scores[np.arange(len(review_scores)), order[:, 0]] - review_scores[
        np.arange(len(review_scores)), order[:, 1]
    ]
    class_error_rate = _history_oof_predicted_class_error_rate(rows, history_idx, schema)
    budget = max(1, int(round(len(review_idx) * 0.20)))
    selected_local = _risk_weighted_low_margin_indices(
        review_pred,
        margins,
        class_error_rate,
        budget,
    )
    selected_texts = [review_texts[i] for i in selected_local]
    selected_labels = [review_labels[i] for i in selected_local]
    selected_errors = sum(review_pred[i] != review_labels[i] for i in selected_local)

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
    retrained_future_f1 = f1_score(future_labels, retrained_future_pred, average="macro", zero_division=0)
    return {
        "protocol": "train_only_history_oof_class_risk_weighted_low_margin_review",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "review_window_count": len(review_idx),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "review_budget_fraction": 0.20,
        "selected_count": len(selected_local),
        "selection": "history_group_safe_3fold_oof_predicted_class_error_rate_weighted_then_low_margin",
        "selection_uses_gold": False,
        "selected_error_rate": round(selected_errors / max(1, len(selected_local)), 6),
        "history_oof_class_error_rate": {
            label: round(float(rate), 6) for label, rate in sorted(class_error_rate.items())
        },
        "baseline_future_macro_f1": round(float(baseline_future_f1), 6),
        "future_macro_f1": round(float(retrained_future_f1), 6),
        "future_macro_f1_gain": round(float(retrained_future_f1 - baseline_future_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_future_pred,
            retrained_future_pred,
        ),
    }


def rolling_history_oof_risk_backtest(rows, schema) -> dict:
    """固定五窗口复验 history-OOF class-risk acquisition；预算固定 20%，不做搜索。"""
    end_fractions = (0.65, 0.75, 0.85, 0.95, 1.0)
    windows = [
        rolling_history_oof_risk_experiment(rows, schema, end_fraction=fraction)
        for fraction in end_fractions
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    return {
        "protocol": "fixed_five_window_train_only_history_oof_class_risk_backtest",
        "external_touched": False,
        "selection_budget_predeclared": 0.20,
        "end_fractions": list(end_fractions),
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


def rolling_recent_history_experiment(rows, schema, *, end_fraction: float) -> dict:
    """固定最近 75% history 的 group-safe 滑窗训练；不读取 review/future Gold 做选择。"""
    history_idx, review_idx, future_idx, dropped_idx, review_cutoff, future_cutoff, end_time = (
        temporal_three_way_group_split_at_end(
            rows,
            end_fraction=end_fraction,
            review_ratio=0.15,
            future_ratio=0.15,
        )
    )
    recent_idx, recent_boundary_dropped, recent_cutoff = recent_group_safe_history_indices(
        rows,
        history_idx,
        keep_fraction=0.75,
    )
    history_texts = [production_like_text(rows[i]) for i in history_idx]
    history_labels = [str(rows[i].event_label) for i in history_idx]
    recent_texts = [production_like_text(rows[i]) for i in recent_idx]
    recent_labels = [str(rows[i].event_label) for i in recent_idx]
    future_texts = [production_like_text(rows[i]) for i in future_idx]
    future_labels = [str(rows[i].event_label) for i in future_idx]
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    baseline = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    candidate = fit_with_descriptions(recent_texts, recent_labels, desc_texts, desc_labels, repeat=1)
    baseline_pred = baseline.predict(future_texts).tolist()
    candidate_pred = candidate.predict(future_texts).tolist()
    baseline_f1 = f1_score(future_labels, baseline_pred, average="macro", zero_division=0)
    candidate_f1 = f1_score(future_labels, candidate_pred, average="macro", zero_division=0)
    return {
        "protocol": "train_only_recent_75pct_history_group_safe_sliding_window",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "recent_history_cutoff_publish_time": recent_cutoff,
        "history_count": len(history_idx),
        "recent_history_count": len(recent_idx),
        "recent_boundary_dropped_count": len(recent_boundary_dropped),
        "review_window_count": len(review_idx),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "selection_uses_gold": False,
        "keep_fraction_predeclared": 0.75,
        "baseline_future_macro_f1": round(float(baseline_f1), 6),
        "future_macro_f1": round(float(candidate_f1), 6),
        "future_macro_f1_gain": round(float(candidate_f1 - baseline_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_pred,
            candidate_pred,
        ),
    }


def rolling_recent_history_backtest(rows, schema) -> dict:
    """固定五窗口验证最近 75% history 滑窗训练；比例预声明，不做 recency 网格。"""
    end_fractions = (0.65, 0.75, 0.85, 0.95, 1.0)
    windows = [
        rolling_recent_history_experiment(rows, schema, end_fraction=fraction)
        for fraction in end_fractions
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    return {
        "protocol": "fixed_five_window_train_only_recent_75pct_history_backtest",
        "external_touched": False,
        "keep_fraction_predeclared": 0.75,
        "end_fractions": list(end_fractions),
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


def rolling_recent_repeat_experiment(rows, schema, *, end_fraction: float) -> dict:
    """保留全部 history，并固定将最新 25% history 额外重复一次；不读取 review/future Gold。"""
    history_idx, review_idx, future_idx, dropped_idx, review_cutoff, future_cutoff, end_time = (
        temporal_three_way_group_split_at_end(
            rows,
            end_fraction=end_fraction,
            review_ratio=0.15,
            future_ratio=0.15,
        )
    )
    repeat_idx = recent_repeat_indices(rows, history_idx, recent_ratio=0.25)
    history_texts = [production_like_text(rows[i]) for i in history_idx]
    history_labels = [str(rows[i].event_label) for i in history_idx]
    repeat_texts = [production_like_text(rows[i]) for i in repeat_idx]
    repeat_labels = [str(rows[i].event_label) for i in repeat_idx]
    future_texts = [production_like_text(rows[i]) for i in future_idx]
    future_labels = [str(rows[i].event_label) for i in future_idx]
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    baseline = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    candidate = fit_with_descriptions(
        history_texts + repeat_texts,
        history_labels + repeat_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    baseline_pred = baseline.predict(future_texts).tolist()
    candidate_pred = candidate.predict(future_texts).tolist()
    baseline_f1 = f1_score(future_labels, baseline_pred, average="macro", zero_division=0)
    candidate_f1 = f1_score(future_labels, candidate_pred, average="macro", zero_division=0)
    return {
        "protocol": "train_only_keep_all_history_repeat_newest_25pct_once",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "repeat_count": len(repeat_idx),
        "review_window_count": len(review_idx),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "selection_uses_gold": False,
        "recent_ratio_predeclared": 0.25,
        "repeat_times_predeclared": 1,
        "baseline_future_macro_f1": round(float(baseline_f1), 6),
        "future_macro_f1": round(float(candidate_f1), 6),
        "future_macro_f1_gain": round(float(candidate_f1 - baseline_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_pred,
            candidate_pred,
        ),
    }


def rolling_recent_repeat_backtest(rows, schema) -> dict:
    """固定五窗口验证“保留全历史 + 最新 25% 重复一次”；比例与重复次数均预声明。"""
    end_fractions = (0.65, 0.75, 0.85, 0.95, 1.0)
    windows = [
        rolling_recent_repeat_experiment(rows, schema, end_fraction=fraction)
        for fraction in end_fractions
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    return {
        "protocol": "fixed_five_window_train_only_keep_all_repeat_newest_25pct_once_backtest",
        "external_touched": False,
        "recent_ratio_predeclared": 0.25,
        "repeat_times_predeclared": 1,
        "end_fractions": list(end_fractions),
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


def rolling_recent_repeat_with_coverage_review_experiment(
    rows,
    schema,
    *,
    end_fraction: float,
) -> dict:
    """固定 50% temporal Gold coverage，再叠加“全历史 + 最新 25% 重复一次”；选样不读 Gold。"""
    history_idx, review_idx, future_idx, dropped_idx, review_cutoff, future_cutoff, end_time = (
        temporal_three_way_group_split_at_end(
            rows,
            end_fraction=end_fraction,
            review_ratio=0.15,
            future_ratio=0.15,
        )
    )
    repeat_idx = recent_repeat_indices(rows, history_idx, recent_ratio=0.25)
    history_texts = [production_like_text(rows[i]) for i in history_idx]
    history_labels = [str(rows[i].event_label) for i in history_idx]
    repeat_texts = [production_like_text(rows[i]) for i in repeat_idx]
    repeat_labels = [str(rows[i].event_label) for i in repeat_idx]
    review_texts = [production_like_text(rows[i]) for i in review_idx]
    review_labels = [str(rows[i].event_label) for i in review_idx]
    future_texts = [production_like_text(rows[i]) for i in future_idx]
    future_labels = [str(rows[i].event_label) for i in future_idx]
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    baseline = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    baseline_future_pred = baseline.predict(future_texts).tolist()
    baseline_future_f1 = f1_score(future_labels, baseline_future_pred, average="macro", zero_division=0)
    review_pred = baseline.predict(review_texts).tolist()
    budget = max(1, int(round(len(review_idx) * 0.50)))
    selected_local = _predicted_class_balanced_temporal_coverage_indices(
        review_pred,
        [rows[i].publish_time for i in review_idx],
        count=budget,
    )
    selected_texts = [review_texts[i] for i in selected_local]
    selected_labels = [review_labels[i] for i in selected_local]

    coverage_only = fit_with_descriptions(
        history_texts + selected_texts,
        history_labels + selected_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    combined = fit_with_descriptions(
        history_texts + repeat_texts + selected_texts,
        history_labels + repeat_labels + selected_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    coverage_pred = coverage_only.predict(future_texts).tolist()
    combined_pred = combined.predict(future_texts).tolist()
    coverage_f1 = f1_score(future_labels, coverage_pred, average="macro", zero_division=0)
    combined_f1 = f1_score(future_labels, combined_pred, average="macro", zero_division=0)
    return {
        "protocol": "train_only_50pct_temporal_gold_coverage_plus_recent25_repeat_once",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "repeat_count": len(repeat_idx),
        "review_window_count": len(review_idx),
        "approved_gold_refresh_count": len(selected_local),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "review_fraction_predeclared": 0.50,
        "recent_ratio_predeclared": 0.25,
        "repeat_times_predeclared": 1,
        "selection": "baseline_predicted_class_balance_plus_within_class_temporal_coverage",
        "selection_uses_review_gold": False,
        "selection_uses_future_gold": False,
        "baseline_future_macro_f1": round(float(baseline_future_f1), 6),
        "coverage_only_future_macro_f1": round(float(coverage_f1), 6),
        "coverage_only_future_macro_f1_gain": round(float(coverage_f1 - baseline_future_f1), 6),
        "combined_future_macro_f1": round(float(combined_f1), 6),
        "combined_future_macro_f1_gain": round(float(combined_f1 - baseline_future_f1), 6),
        "combined_incremental_over_coverage": round(float(combined_f1 - coverage_f1), 6),
        "combined_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_future_pred,
            combined_pred,
        ),
        "incremental_over_coverage_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            coverage_pred,
            combined_pred,
        ),
    }


def rolling_recent_repeat_with_coverage_review_backtest(rows, schema) -> dict:
    """固定五窗口验证 50% Gold coverage + 最新 25% 重复一次，判断能否低成本达到 harmed=0。"""
    end_fractions = (0.65, 0.75, 0.85, 0.95, 1.0)
    windows = [
        rolling_recent_repeat_with_coverage_review_experiment(rows, schema, end_fraction=fraction)
        for fraction in end_fractions
    ]
    gains = [window["combined_future_macro_f1_gain"] for window in windows]
    incremental = [window["combined_incremental_over_coverage"] for window in windows]
    return {
        "protocol": "fixed_five_window_train_only_50pct_gold_coverage_plus_recent25_repeat_once_backtest",
        "external_touched": False,
        "review_fraction_predeclared": 0.50,
        "recent_ratio_predeclared": 0.25,
        "repeat_times_predeclared": 1,
        "end_fractions": list(end_fractions),
        "windows": windows,
        "summary": {
            "window_count": len(gains),
            "mean_combined_future_macro_f1_gain": round(float(np.mean(gains)), 6),
            "median_combined_future_macro_f1_gain": round(float(np.median(gains)), 6),
            "min_combined_future_macro_f1_gain": round(float(np.min(gains)), 6),
            "max_combined_future_macro_f1_gain": round(float(np.max(gains)), 6),
            "positive_windows": int(sum(gain > 0 for gain in gains)),
            "all_windows_positive": bool(all(gain > 0 for gain in gains)),
            "mean_incremental_over_coverage": round(float(np.mean(incremental)), 6),
            "positive_incremental_windows": int(sum(gain > 0 for gain in incremental)),
        },
    }


def _review_group_representative_propagation(rows, review_idx: list[int]) -> tuple[list[int], list[str]]:
    """每个 duplication group 只审最早一篇，并把该 Gold 标签传播到同组；不读取同组其余标签做决策。"""
    groups = duplication_groups(rows)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index in review_idx:
        by_group[groups[index]].append(index)
    representative_idx: list[int] = []
    propagated_labels: list[str] = []
    for group in sorted(by_group):
        indices = sorted(by_group[group], key=lambda i: (rows[i].publish_time, str(rows[i].article_id)))
        representative = indices[0]
        representative_idx.append(representative)
        label = str(rows[representative].event_label)
        propagated_labels.extend([label] * len(indices))
    return representative_idx, propagated_labels


def rolling_group_representative_gold_refresh_experiment(rows, schema, *, end_fraction: float) -> dict:
    """固定五窗口候选：每个 review duplication group 仅人工审批 1 篇代表文，再向组内传播 Gold。"""
    history_idx, review_idx, future_idx, dropped_idx, review_cutoff, future_cutoff, end_time = (
        temporal_three_way_group_split_at_end(
            rows,
            end_fraction=end_fraction,
            review_ratio=0.15,
            future_ratio=0.15,
        )
    )
    history_texts = [production_like_text(rows[i]) for i in history_idx]
    history_labels = [str(rows[i].event_label) for i in history_idx]
    future_texts = [production_like_text(rows[i]) for i in future_idx]
    future_labels = [str(rows[i].event_label) for i in future_idx]
    desc_texts, desc_labels = label_description_examples(schema, scope="company")

    groups = duplication_groups(rows)
    review_by_group: dict[str, list[int]] = defaultdict(list)
    for index in review_idx:
        review_by_group[groups[index]].append(index)
    representative_idx: list[int] = []
    propagated_by_index: dict[int, str] = {}
    inconsistent_rows = 0
    inconsistent_groups = 0
    for group in sorted(review_by_group):
        indices = sorted(review_by_group[group], key=lambda i: (rows[i].publish_time, str(rows[i].article_id)))
        representative = indices[0]
        representative_idx.append(representative)
        approved_label = str(rows[representative].event_label)
        group_inconsistent = False
        for index in indices:
            propagated_by_index[index] = approved_label
            if str(rows[index].event_label) != approved_label:
                inconsistent_rows += 1
                group_inconsistent = True
        inconsistent_groups += int(group_inconsistent)

    review_texts = [production_like_text(rows[i]) for i in review_idx]
    propagated_labels = [propagated_by_index[i] for i in review_idx]
    baseline = fit_with_descriptions(history_texts, history_labels, desc_texts, desc_labels, repeat=1)
    candidate = fit_with_descriptions(
        history_texts + review_texts,
        history_labels + propagated_labels,
        desc_texts,
        desc_labels,
        repeat=1,
    )
    baseline_pred = baseline.predict(future_texts).tolist()
    candidate_pred = candidate.predict(future_texts).tolist()
    baseline_f1 = f1_score(future_labels, baseline_pred, average="macro", zero_division=0)
    candidate_f1 = f1_score(future_labels, candidate_pred, average="macro", zero_division=0)
    manual_actions = len(representative_idx)
    row_count = len(review_idx)
    return {
        "protocol": "train_only_duplication_group_one_representative_gold_propagation",
        "external_touched": False,
        "end_fraction": end_fraction,
        "window_end_publish_time": end_time,
        "review_cutoff_publish_time": review_cutoff,
        "future_cutoff_publish_time": future_cutoff,
        "history_count": len(history_idx),
        "review_window_count": row_count,
        "review_duplication_group_count": len(review_by_group),
        "manual_review_actions": manual_actions,
        "manual_action_fraction_vs_full_review": round(manual_actions / max(1, row_count), 6),
        "future_count": len(future_idx),
        "dropped_boundary_count": len(dropped_idx),
        "selection": "earliest_article_per_duplication_group_then_blind_group_label_propagation",
        "selection_uses_future_gold": False,
        "propagation_uses_nonrepresentative_gold": False,
        "audit_only_inconsistent_group_count": inconsistent_groups,
        "audit_only_inconsistent_row_count": inconsistent_rows,
        "audit_only_propagation_precision": round(1.0 - inconsistent_rows / max(1, row_count), 6),
        "baseline_future_macro_f1": round(float(baseline_f1), 6),
        "future_macro_f1": round(float(candidate_f1), 6),
        "future_macro_f1_gain": round(float(candidate_f1 - baseline_f1), 6),
        "future_gain_bootstrap": paired_bootstrap_macro_f1_gain(
            future_labels,
            baseline_pred,
            candidate_pred,
        ),
    }


def rolling_group_representative_gold_refresh_backtest(rows, schema) -> dict:
    """固定五窗口验证 duplication-group 单代表审批传播，量化人工成本与 harmed 风险。"""
    end_fractions = (0.65, 0.75, 0.85, 0.95, 1.0)
    windows = [
        rolling_group_representative_gold_refresh_experiment(rows, schema, end_fraction=fraction)
        for fraction in end_fractions
    ]
    gains = [window["future_macro_f1_gain"] for window in windows]
    action_fractions = [window["manual_action_fraction_vs_full_review"] for window in windows]
    precisions = [window["audit_only_propagation_precision"] for window in windows]
    return {
        "protocol": "fixed_five_window_train_only_duplication_group_representative_gold_refresh_backtest",
        "external_touched": False,
        "end_fractions": list(end_fractions),
        "windows": windows,
        "summary": {
            "window_count": len(gains),
            "mean_future_macro_f1_gain": round(float(np.mean(gains)), 6),
            "median_future_macro_f1_gain": round(float(np.median(gains)), 6),
            "min_future_macro_f1_gain": round(float(np.min(gains)), 6),
            "max_future_macro_f1_gain": round(float(np.max(gains)), 6),
            "positive_windows": int(sum(gain > 0 for gain in gains)),
            "all_windows_positive": bool(all(gain > 0 for gain in gains)),
            "mean_manual_action_fraction_vs_full_review": round(float(np.mean(action_fractions)), 6),
            "mean_audit_only_propagation_precision": round(float(np.mean(precisions)), 6),
            "min_audit_only_propagation_precision": round(float(np.min(precisions)), 6),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    rows = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    train_idx, val_idx, dropped_idx, cutoff = temporal_group_split(rows, holdout_ratio=args.holdout_ratio)
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    train_labels = [str(row.event_label) for row in train_rows]
    val_labels = [str(row.event_label) for row in val_rows]
    train_texts = [production_like_text(row) for row in train_rows]
    val_texts = [production_like_text(row) for row in val_rows]

    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    desc_texts, desc_labels = label_description_examples(schema, scope="company")
    baseline_metrics = evaluate_model(
        train_texts, train_labels, val_texts, val_labels, desc_texts, desc_labels
    )
    repeat_idx = recent_repeat_indices(rows, train_idx, recent_ratio=0.25)
    repeat_texts = [production_like_text(rows[i]) for i in repeat_idx]
    repeat_labels = [str(rows[i].event_label) for i in repeat_idx]
    recent_metrics = evaluate_model(
        train_texts + repeat_texts,
        train_labels + repeat_labels,
        val_texts,
        val_labels,
        desc_texts,
        desc_labels,
    )
    train_classes = set(train_labels)
    unseen_truth = sorted(set(val_labels) - train_classes)
    result = {
        "protocol": "train_only_temporal_duplication_safe_holdout",
        "external_touched": False,
        "holdout_ratio": args.holdout_ratio,
        "cutoff_publish_time": cutoff,
        "train_count": len(train_idx),
        "holdout_count": len(val_idx),
        "dropped_boundary_count": len(dropped_idx),
        "train_group_count": len({duplication_groups(rows)[i] for i in train_idx}),
        "holdout_group_count": len({duplication_groups(rows)[i] for i in val_idx}),
        "group_overlap": 0,
        "train_max_publish_time": max(rows[i].publish_time for i in train_idx).isoformat(),
        "holdout_min_publish_time": min(rows[i].publish_time for i in val_idx).isoformat(),
        "unseen_holdout_labels": unseen_truth,
        "metrics": baseline_metrics,
        "recency_robustness": {
            "protocol": "repeat_newest_25pct_train_once",
            "selection_inputs": "train publish_time only; validation labels used for evaluation only",
            "repeat_count": len(repeat_idx),
            "metrics": recent_metrics,
            "macro_f1_gain": round(
                recent_metrics["macro_f1"] - baseline_metrics["macro_f1"], 6
            ),
        },
        "rolling_reviewed_gold": rolling_review_tranche_experiment(rows, schema),
        "rolling_review_backtest": rolling_review_backtest(rows, schema),
        "rolling_snapshot_disagreement_backtest": rolling_snapshot_disagreement_backtest(rows, schema),
        "rolling_history_ensemble_disagreement_backtest": rolling_history_ensemble_disagreement_backtest(rows, schema),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
