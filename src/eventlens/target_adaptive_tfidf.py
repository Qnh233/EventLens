from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


@dataclass
class TargetAdaptiveTfidfModel:
    vectorizer: TfidfVectorizer
    classifier: LinearSVC


def fit_target_adaptive_svc(
    labeled_texts: list[str],
    labels: list[str],
    *,
    domain_texts: list[str],
    description_texts: list[str],
    description_labels: list[str],
    random_state: int = 42,
) -> TargetAdaptiveTfidfModel:
    """无标签目标域文本只学习词表/IDF；监督边界只由 Gold + Schema 描述学习。"""

    if len(labeled_texts) != len(labels):
        raise ValueError("labeled text/label count mismatch")
    if len(description_texts) != len(description_labels):
        raise ValueError("description text/label count mismatch")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        max_features=80000,
        sublinear_tf=True,
    )
    vectorizer.fit(domain_texts + labeled_texts + description_texts)
    train_texts = labeled_texts + description_texts
    train_labels = labels + description_labels
    classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=random_state)
    classifier.fit(vectorizer.transform(train_texts), train_labels)
    return TargetAdaptiveTfidfModel(vectorizer=vectorizer, classifier=classifier)


def decision_scores(
    model: TargetAdaptiveTfidfModel,
    texts: list[str],
    classes: list[str],
) -> np.ndarray:
    local_scores = np.asarray(
        model.classifier.decision_function(model.vectorizer.transform(texts)),
        dtype=np.float64,
    )
    local_classes = [str(value) for value in model.classifier.classes_]
    output = np.full((len(texts), len(classes)), -1e9, dtype=np.float64)
    class_to_index = {label: index for index, label in enumerate(classes)}
    for local_index, label in enumerate(local_classes):
        output[:, class_to_index[label]] = local_scores[:, local_index]
    return output
