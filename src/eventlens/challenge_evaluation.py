from __future__ import annotations

from collections import Counter, defaultdict

from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score

from eventlens.schema import ArticleRecord
from eventlens.subject_routing import SubjectRouteResult


class ChallengeSliceMetrics(BaseModel):
    sample_count: int
    accuracy: float
    macro_f1: float


def evaluate_challenge_slices(
    train_articles: list[ArticleRecord],
    test_articles: list[ArticleRecord],
    predictions: list[str],
    routes: list[SubjectRouteResult],
    *,
    scope: str,
    rare_event_max_train_count: int,
    long_tail_source_max_train_count: int,
    long_text_percentile: float,
) -> dict[str, ChallengeSliceMetrics]:
    if len(test_articles) != len(predictions) or len(test_articles) != len(routes):
        raise ValueError("challenge slice 输入数量不一致")
    label_counts = Counter(str(row.event_label or "") for row in train_articles)
    source_counts = Counter(row.source for row in train_articles if row.source)
    subject_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train_articles:
        subject = _subject(row, scope)
        if subject and row.event_label:
            subject_labels[subject][str(row.event_label)] += 1
    subject_prior = {
        subject: counts.most_common(1)[0][0]
        for subject, counts in subject_labels.items()
        if counts
    }
    train_lengths = sorted(_text_length(row) for row in train_articles)
    long_threshold = _quantile(train_lengths, long_text_percentile)

    slices: dict[str, list[int]] = {
        "all": list(range(len(test_articles))),
        "anti_subject_prior": [],
        "subject_unseen": [],
        "ambiguous_subject": [],
        "rare_event": [],
        "long_tail_source": [],
        "long_text": [],
    }
    for index, (article, route) in enumerate(zip(test_articles, routes)):
        truth = str(article.event_label or "")
        subject = _subject(article, scope)
        if subject and subject in subject_prior and subject_prior[subject] != truth:
            slices["anti_subject_prior"].append(index)
        if subject and subject not in subject_prior:
            slices["subject_unseen"].append(index)
        if route.accepted_subject_code is None and len(route.candidates) > 1:
            slices["ambiguous_subject"].append(index)
        if label_counts[truth] <= rare_event_max_train_count:
            slices["rare_event"].append(index)
        if article.source and source_counts[article.source] <= long_tail_source_max_train_count:
            slices["long_tail_source"].append(index)
        if _text_length(article) >= long_threshold:
            slices["long_text"].append(index)

    return {
        name: _metrics(test_articles, predictions, indices)
        for name, indices in slices.items()
    }


def _metrics(
    articles: list[ArticleRecord], predictions: list[str], indices: list[int]
) -> ChallengeSliceMetrics:
    if not indices:
        return ChallengeSliceMetrics(sample_count=0, accuracy=0.0, macro_f1=0.0)
    truth = [str(articles[index].event_label) for index in indices]
    pred = [predictions[index] for index in indices]
    return ChallengeSliceMetrics(
        sample_count=len(indices),
        accuracy=round(accuracy_score(truth, pred), 6),
        macro_f1=round(f1_score(truth, pred, average="macro", zero_division=0), 6),
    )


def _subject(article: ArticleRecord, scope: str) -> str:
    return article.trading_code if scope == "company" else article.industry_code


def _text_length(article: ArticleRecord) -> int:
    return len(article.title or "") + len(article.content or "")


def _quantile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("long_text_percentile 必须位于 [0,1]")
    index = min(len(values) - 1, int((len(values) - 1) * percentile))
    return values[index]
