from __future__ import annotations

import argparse
from collections import Counter

from sklearn.metrics import classification_report

from benchmark_classical_event import Candidate, build_svc
from eventlens.config import load_settings
from eventlens.io import read_competition_labeled_excel
from eventlens.preprocess import build_model_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    parser.add_argument("--top-confusions", type=int, default=30)
    args = parser.parse_args()

    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)[f"{args.scope}_event"]
    test = read_competition_labeled_excel(settings.paths.tagged_test)[f"{args.scope}_event"]
    max_chars = settings.model.text.max_content_chars
    train_texts = [build_model_text(row, max_chars) for row in train]
    test_texts = [build_model_text(row, max_chars) for row in test]
    train_labels = [str(row.event_label) for row in train]
    test_labels = [str(row.event_label) for row in test]

    model = build_svc(Candidate("diagnostic", no_subject=False, char_word=False))
    model.fit(train_texts, train_labels)
    pred = [str(value) for value in model.predict(test_texts)]
    report = classification_report(test_labels, pred, output_dict=True, zero_division=0)
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    print("label\ttrain\ttest\tprecision\trecall\tf1")
    for label in sorted(test_counts, key=lambda name: (report[name]["f1-score"], test_counts[name], name)):
        row = report[label]
        print(
            f"{label}\t{train_counts[label]}\t{test_counts[label]}\t"
            f"{row['precision']:.3f}\t{row['recall']:.3f}\t{row['f1-score']:.3f}"
        )

    print("\nTOP_CONFUSIONS")
    confusions = Counter((truth, guess) for truth, guess in zip(test_labels, pred) if truth != guess)
    for (truth, guess), count in confusions.most_common(args.top_confusions):
        print(f"{count}\t{truth}\t->\t{guess}")


if __name__ == "__main__":
    main()
