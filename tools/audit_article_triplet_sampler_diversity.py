from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from benchmark_article_triplet_bge import duplication_groups
from eventlens.config import load_settings
from eventlens.io import read_competition_labeled_excel


def sampler_stats(labels: list[str], exponent: float, *, epochs: int = 3) -> dict[str, float]:
    counts = Counter(labels)
    weights = np.asarray([counts[label] ** (-exponent) for label in labels], dtype=np.float64)
    probability = weights / weights.sum()
    draws = len(labels) * epochs
    expected_unique = np.mean(1.0 - np.power(1.0 - probability, draws))
    ess = 1.0 / np.square(probability).sum()
    expected_draws = draws * probability
    return {
        "expected_unique_anchor_fraction_3epochs": round(float(expected_unique), 4),
        "max_expected_draws_per_anchor_3epochs": round(float(expected_draws.max()), 3),
        "min_expected_draws_per_anchor_3epochs": round(float(expected_draws.min()), 3),
        "sampling_ess": round(float(ess), 1),
        "ess_fraction": round(float(ess / len(labels)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)["company_event"]
    labels = [str(row.event_label) for row in train]
    groups = duplication_groups(train)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.random_state)
    row_ids = np.arange(len(labels))
    folds: list[dict[str, object]] = []
    for fold, (fit_idx, _) in enumerate(splitter.split(row_ids, labels, groups), start=1):
        fit_labels = [labels[index] for index in fit_idx]
        folds.append(
            {
                "fold": fold,
                "fit_samples": len(fit_labels),
                "classes": len(set(fit_labels)),
                "uniform": sampler_stats(fit_labels, 0.0),
                "sqrt_inverse": sampler_stats(fit_labels, 0.5),
                "full_inverse": sampler_stats(fit_labels, 1.0),
            }
        )

    payload = {
        "protocol": "train-only analytical sampler diversity audit on seed-42 duplication-safe folds; 3 epochs",
        "external_touched": False,
        "folds": folds,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
