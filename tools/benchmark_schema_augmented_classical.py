from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark_classical_event import Candidate, build_svc, metrics
from benchmark_schema_constrained_svc import label_description_examples
from eventlens.config import load_settings
from eventlens.event_retrieval import EventSchemaIndex
from eventlens.io import read_competition_labeled_excel
from eventlens.preprocess import build_model_text


DESCRIPTION_REPEATS = (0, 1, 2, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)[f"{args.scope}_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)[f"{args.scope}_event"]
    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    description_texts, description_labels = label_description_examples(schema, scope=args.scope)
    max_chars = settings.model.text.max_content_chars
    train_texts = [build_model_text(row, max_chars) for row in train]
    test_texts = [build_model_text(row, max_chars) for row in test]
    labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)

    candidate = Candidate("schema_augmented_char_subject", no_subject=False, char_word=False)
    oof: dict[str, dict[str, float | list[float]]] = {}
    selected_repeat = 0
    selected_score = -1.0
    for repeat in DESCRIPTION_REPEATS:
        fold_scores: list[float] = []
        for train_idx, val_idx in folds.split(train_texts, labels):
            model = build_svc(candidate)
            model.fit(
                [train_texts[i] for i in train_idx] + description_texts * repeat,
                [labels[i] for i in train_idx] + description_labels * repeat,
            )
            pred = [str(value) for value in model.predict([train_texts[i] for i in val_idx])]
            fold_scores.append(metrics([labels[i] for i in val_idx], pred)["macro_f1"])
        mean_score = float(np.mean(fold_scores))
        oof[str(repeat)] = {
            "macro_f1_mean": round(mean_score, 6),
            "macro_f1_std": round(float(np.std(fold_scores)), 6),
            "folds": fold_scores,
        }
        if mean_score > selected_score:
            selected_repeat = repeat
            selected_score = mean_score

    model = build_svc(candidate)
    model.fit(
        train_texts + description_texts * selected_repeat,
        labels + description_labels * selected_repeat,
    )
    test_pred = [str(value) for value in model.predict(test_texts)]
    external = metrics(test_labels, test_pred)
    payload = {
        "scope": args.scope,
        "protocol": "schema descriptions are train-time augmentation only; repeat selected by 3-fold train OOF; tagged test evaluated once",
        "label_description_count": len(description_texts),
        "selected_description_repeat": selected_repeat,
        "selected_oof": oof[str(selected_repeat)],
        "external": external,
        "gate_macro_f1": 0.80,
        "gate_passed": external["macro_f1"] >= 0.80,
        "oof": oof,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
