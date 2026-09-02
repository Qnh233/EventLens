from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def bootstrap_classification_ci(
    truth: list[str],
    predictions: list[str],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, object]:
    """对固定预测做样本级 bootstrap；仅衡量评估波动，不参与模型选择。"""
    if len(truth) != len(predictions):
        raise ValueError("truth/prediction count mismatch")
    if not truth:
        raise ValueError("truth must not be empty")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    truth_array = np.asarray(truth, dtype=object)
    pred_array = np.asarray(predictions, dtype=object)
    rng = np.random.default_rng(seed)
    accuracy_samples = np.empty(n_bootstrap, dtype=np.float64)
    macro_f1_samples = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sampled = rng.integers(0, len(truth_array), size=len(truth_array))
        sampled_truth = truth_array[sampled]
        sampled_pred = pred_array[sampled]
        accuracy_samples[index] = accuracy_score(sampled_truth, sampled_pred)
        macro_f1_samples[index] = f1_score(
            sampled_truth,
            sampled_pred,
            average="macro",
            zero_division=0,
        )

    alpha = (1.0 - confidence) / 2.0

    def summarize(values: np.ndarray, point: float) -> dict[str, float]:
        return {
            "point": round(float(point), 6),
            "lower": round(float(np.quantile(values, alpha)), 6),
            "upper": round(float(np.quantile(values, 1.0 - alpha)), 6),
            "std": round(float(np.std(values, ddof=1)), 6),
        }

    return {
        "method": "sample bootstrap on fixed predictions",
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "seed": seed,
        "accuracy": summarize(
            accuracy_samples,
            accuracy_score(truth_array, pred_array),
        ),
        "macro_f1": summarize(
            macro_f1_samples,
            f1_score(truth_array, pred_array, average="macro", zero_division=0),
        ),
    }


def margin_error_profile(
    margins: list[float] | np.ndarray,
    is_error: list[bool] | np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> list[dict[str, float | int]]:
    values = np.asarray(margins, dtype=np.float64)
    errors = np.asarray(is_error, dtype=bool)
    if values.ndim != 1 or values.shape != errors.shape:
        raise ValueError("margin/error shape mismatch")
    total_errors = int(errors.sum())
    output: list[dict[str, float | int]] = []
    for quantile in quantiles:
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantiles must be in (0, 1]")
        threshold = float(np.quantile(values, quantile))
        selected = values <= threshold
        selected_count = int(selected.sum())
        selected_errors = int((selected & errors).sum())
        output.append(
            {
                "quantile": float(quantile),
                "threshold": round(threshold, 6),
                "selected_count": selected_count,
                "error_rate": round(selected_errors / max(1, selected_count), 6),
                "error_coverage": round(selected_errors / max(1, total_errors), 6),
            }
        )
    return output


def candidate_error_coverage(
    truth: list[str],
    predictions: list[str],
    candidate_groups: list[list[str]],
) -> dict[str, float | int]:
    if not (len(truth) == len(predictions) == len(candidate_groups)):
        raise ValueError("truth/prediction/candidate count mismatch")
    error_indices = [i for i, (gold, pred) in enumerate(zip(truth, predictions)) if gold != pred]
    covered = sum(truth[i] in candidate_groups[i] for i in error_indices)
    return {
        "error_count": len(error_indices),
        "covered_error_count": int(covered),
        "error_candidate_coverage": round(covered / max(1, len(error_indices)), 6),
    }


def confusion_concentration(
    truth: list[str],
    predictions: list[str],
    *,
    top_n: int = 10,
) -> dict[str, object]:
    if len(truth) != len(predictions):
        raise ValueError("truth/prediction count mismatch")
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    counter = Counter(
        (gold, pred)
        for gold, pred in zip(truth, predictions)
        if gold != pred
    )
    total_errors = sum(counter.values())
    top_pairs = [
        {"truth": gold, "prediction": pred, "count": count}
        for (gold, pred), count in counter.most_common(top_n)
    ]
    top_error_count = sum(row["count"] for row in top_pairs)
    return {
        "error_count": total_errors,
        "unique_confusion_pairs": len(counter),
        "top_n": top_n,
        "top_n_error_share": round(top_error_count / max(1, total_errors), 6),
        "pairs": top_pairs,
    }


def hardcase_oracle_frontier(
    truth: list[str],
    predictions: list[str],
    margins: list[float] | np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
) -> list[dict[str, float | int]]:
    """量化低 margin hard-case 若由完美专家纠正时的上限，不作为生产指标。"""
    if len(truth) != len(predictions):
        raise ValueError("truth/prediction count mismatch")
    values = np.asarray(margins, dtype=np.float64)
    if values.shape != (len(truth),):
        raise ValueError("margin count mismatch")

    baseline_f1 = f1_score(truth, predictions, average="macro", zero_division=0)
    output: list[dict[str, float | int]] = []
    for quantile in quantiles:
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantiles must be in (0, 1]")
        threshold = float(np.quantile(values, quantile))
        selected = values <= threshold
        oracle = list(predictions)
        corrected = 0
        for index, is_selected in enumerate(selected):
            if is_selected and oracle[index] != truth[index]:
                oracle[index] = truth[index]
                corrected += 1
        oracle_f1 = f1_score(truth, oracle, average="macro", zero_division=0)
        output.append(
            {
                "quantile": float(quantile),
                "threshold": round(threshold, 6),
                "selected_count": int(selected.sum()),
                "selected_fraction": round(float(selected.mean()), 6),
                "correctable_errors": corrected,
                "oracle_accuracy": round(float(accuracy_score(truth, oracle)), 6),
                "oracle_macro_f1": round(float(oracle_f1), 6),
                "oracle_macro_f1_gain": round(float(oracle_f1 - baseline_f1), 6),
            }
        )
    return output
