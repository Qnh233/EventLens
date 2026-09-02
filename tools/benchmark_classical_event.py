from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from eventlens.baseline import train_baseline
from eventlens.config import load_settings
from eventlens.io import read_competition_labeled_excel
from eventlens.preprocess import build_model_text, clean_text


@dataclass(frozen=True)
class Candidate:
    name: str
    no_subject: bool
    char_word: bool


def text_without_subject(article, max_content_chars: int) -> str:
    parts = [article.title, article.source, article.content[:max_content_chars]]
    return " ".join(clean_text(part) for part in parts if clean_text(part))


def build_svc(candidate: Candidate) -> Pipeline:
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=80000 if not candidate.char_word else 60000,
        sublinear_tf=True,
    )
    features = char
    if candidate.char_word:
        features = FeatureUnion(
            [
                ("char", char),
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=30000,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("tfidf", features),
            ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=42)),
        ]
    )


def metrics(truth: list[str], pred: list[str]) -> dict[str, float]:
    return {
        "accuracy": round(accuracy_score(truth, pred), 6),
        "macro_f1": round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)[f"{args.scope}_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)[f"{args.scope}_event"]
    max_chars = settings.model.text.max_content_chars
    labels = [str(row.event_label) for row in train]
    candidates = [
        Candidate("linear_svc_char_subject", no_subject=False, char_word=False),
        Candidate("linear_svc_char_no_subject", no_subject=True, char_word=False),
        Candidate("linear_svc_char_word_no_subject", no_subject=True, char_word=True),
    ]
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.model.random_state)
    candidate_rows = []
    for candidate in candidates:
        texts = [
            text_without_subject(row, max_chars)
            if candidate.no_subject
            else build_model_text(row, max_chars)
            for row in train
        ]
        fold_scores = []
        for train_index, validation_index in folds.split(texts, labels):
            model = build_svc(candidate)
            model.fit([texts[i] for i in train_index], [labels[i] for i in train_index])
            pred = [str(x) for x in model.predict([texts[i] for i in validation_index])]
            fold_scores.append(
                metrics([labels[i] for i in validation_index], pred)["macro_f1"]
            )
        candidate_rows.append(
            {
                "name": candidate.name,
                "cv_macro_f1_mean": round(float(np.mean(fold_scores)), 6),
                "cv_macro_f1_std": round(float(np.std(fold_scores)), 6),
                "folds": fold_scores,
            }
        )

    winner_row = max(candidate_rows, key=lambda row: row["cv_macro_f1_mean"])
    winner = next(row for row in candidates if row.name == winner_row["name"])
    train_texts = [
        text_without_subject(row, max_chars)
        if winner.no_subject
        else build_model_text(row, max_chars)
        for row in train
    ]
    test_texts = [
        text_without_subject(row, max_chars)
        if winner.no_subject
        else build_model_text(row, max_chars)
        for row in test
    ]
    winner_model = build_svc(winner)
    winner_model.fit(train_texts, labels)
    winner_pred = [str(x) for x in winner_model.predict(test_texts)]
    winner_external = metrics([str(row.event_label) for row in test], winner_pred)

    baseline_model = train_baseline(
        [row.model_copy(update={"polarity_label": None}) for row in train],
        settings.model.model_dump(),
    )
    baseline_pred = [baseline_model.predict_one(row).event_type for row in test]
    baseline_external = metrics([str(row.event_label) for row in test], baseline_pred)
    payload = {
        "scope": args.scope,
        "selection_protocol": "3-fold stratified CV on tagged train; official tagged test used once for winner",
        "candidates": candidate_rows,
        "winner": winner.name,
        "baseline_external": baseline_external,
        "winner_external": winner_external,
        "macro_f1_gain": round(
            winner_external["macro_f1"] - baseline_external["macro_f1"], 6
        ),
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
