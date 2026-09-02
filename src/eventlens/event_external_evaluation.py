from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from eventlens.event_retrieval import RoutedArticleRecallResult
from eventlens.challenge_evaluation import ChallengeSliceMetrics, evaluate_challenge_slices
from eventlens.schema import ArticleRecord
from eventlens.subject_routing import SubjectRouteResult


class ClassificationMetrics(BaseModel):
    sample_count: int
    accuracy: float
    macro_f1: float


class SubjectRoutingMetrics(BaseModel):
    sample_count: int
    hard_route_count: int
    hard_route_coverage: float
    hard_route_precision: float
    effective_hit_at_1: float
    effective_hit_at_k: float


class EventRetrievalMetrics(BaseModel):
    sample_count: int
    with_candidates: int
    hit_at_1: float
    hit_at_k: float
    top1_macro_f1: float


class ExternalEventEvaluationReport(BaseModel):
    scope: str
    top_k: int
    baseline: ClassificationMetrics
    embedding_linear: ClassificationMetrics | None = None
    candidate_guard: ClassificationMetrics
    subject_routing: SubjectRoutingMetrics
    routed_event_retrieval: EventRetrievalMetrics
    challenge_slices: dict[str, ChallengeSliceMetrics] = Field(default_factory=dict)


def evaluate_external_event_results(
    articles: list[ArticleRecord],
    baseline_predictions: list[str],
    routes: list[SubjectRouteResult],
    recalls: list[RoutedArticleRecallResult],
    *,
    scope: str,
    top_k: int,
    embedding_linear_predictions: list[str] | None = None,
    train_articles: list[ArticleRecord] | None = None,
    challenge_config: dict | None = None,
) -> ExternalEventEvaluationReport:
    if not (
        len(articles) == len(baseline_predictions) == len(routes) == len(recalls)
    ):
        raise ValueError("外部事件评测输入数量不一致")
    if not articles:
        raise ValueError("外部事件评测至少需要 1 条样本")

    article_ids = [row.article_id for row in articles]
    if article_ids != [row.article_id for row in routes]:
        raise ValueError("主体路由顺序与评测文章不一致")
    if article_ids != [row.article_id for row in recalls]:
        raise ValueError("事件召回顺序与评测文章不一致")

    y_true = [str(row.event_label or "") for row in articles]
    if any(not label for label in y_true):
        raise ValueError("外部事件评测要求每篇文章都有 event_label")

    baseline = _classification_metrics(y_true, baseline_predictions)
    embedding_linear = (
        _classification_metrics(y_true, embedding_linear_predictions)
        if embedding_linear_predictions is not None
        else None
    )
    guarded_predictions = _candidate_guard_predictions(
        baseline_predictions, recalls, top_k=top_k
    )
    candidate_guard = _classification_metrics(y_true, guarded_predictions)
    routing = _routing_metrics(articles, routes, scope=scope)
    retrieval = _retrieval_metrics(y_true, recalls, top_k=top_k)
    challenge_slices = {}
    if train_articles is not None and challenge_config is not None:
        challenge_slices = evaluate_challenge_slices(
            train_articles,
            articles,
            baseline_predictions,
            routes,
            scope=scope,
            rare_event_max_train_count=int(challenge_config["rare_event_max_train_count"]),
            long_tail_source_max_train_count=int(
                challenge_config["long_tail_source_max_train_count"]
            ),
            long_text_percentile=float(challenge_config["long_text_percentile"]),
        )
    return ExternalEventEvaluationReport(
        scope=scope,
        top_k=top_k,
        baseline=baseline,
        embedding_linear=embedding_linear,
        candidate_guard=candidate_guard,
        subject_routing=routing,
        routed_event_retrieval=retrieval,
        challenge_slices=challenge_slices,
    )


def fit_embedding_linear_predictions(
    train_vectors: np.ndarray,
    train_labels: list[str],
    test_vectors: np.ndarray,
    *,
    random_state: int = 42,
) -> list[str]:
    train = np.asarray(train_vectors, dtype=np.float32)
    test = np.asarray(test_vectors, dtype=np.float32)
    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("embedding 线性分类要求二维向量")
    if train.shape[0] != len(train_labels):
        raise ValueError("训练 embedding 数量与标签不一致")
    if train.shape[1] != test.shape[1]:
        raise ValueError("训练/测试 embedding 维度不一致")
    classifier = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
    )
    classifier.fit(train, train_labels)
    return [str(label) for label in classifier.predict(test)]


def _classification_metrics(
    y_true: list[str], y_pred: list[str]
) -> ClassificationMetrics:
    return ClassificationMetrics(
        sample_count=len(y_true),
        accuracy=round(accuracy_score(y_true, y_pred), 6),
        macro_f1=round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 6
        ),
    )


def _routing_metrics(
    articles: list[ArticleRecord],
    routes: list[SubjectRouteResult],
    *,
    scope: str,
) -> SubjectRoutingMetrics:
    hard_route_count = 0
    hard_route_correct = 0
    hit_1 = 0
    hit_k = 0
    for article, route in zip(articles, routes):
        truth = article.trading_code if scope == "company" else article.industry_code
        effective_codes = (
            [route.accepted_subject_code]
            if route.accepted_subject_code
            else [candidate.subject_code for candidate in route.candidates]
        )
        hard_route_count += int(route.accepted_subject_code is not None)
        hard_route_correct += int(
            route.accepted_subject_code is not None
            and route.accepted_subject_code == truth
        )
        hit_1 += int(bool(effective_codes) and effective_codes[0] == truth)
        hit_k += int(truth in effective_codes)

    count = len(articles)
    return SubjectRoutingMetrics(
        sample_count=count,
        hard_route_count=hard_route_count,
        hard_route_coverage=round(hard_route_count / count, 6),
        hard_route_precision=round(
            hard_route_correct / max(1, hard_route_count), 6
        ),
        effective_hit_at_1=round(hit_1 / count, 6),
        effective_hit_at_k=round(hit_k / count, 6),
    )


def _retrieval_metrics(
    y_true: list[str],
    recalls: list[RoutedArticleRecallResult],
    *,
    top_k: int,
) -> EventRetrievalMetrics:
    with_candidates = 0
    hit_1 = 0
    hit_k = 0
    top1_predictions: list[str] = []
    for truth, recall in zip(y_true, recalls):
        names = [candidate.event_name for candidate in recall.candidates[:top_k]]
        with_candidates += int(bool(names))
        hit_1 += int(bool(names) and names[0] == truth)
        hit_k += int(truth in names)
        top1_predictions.append(names[0] if names else "__NO_CANDIDATE__")

    count = len(y_true)
    return EventRetrievalMetrics(
        sample_count=count,
        with_candidates=with_candidates,
        hit_at_1=round(hit_1 / count, 6),
        hit_at_k=round(hit_k / count, 6),
        top1_macro_f1=round(
            f1_score(y_true, top1_predictions, average="macro", zero_division=0), 6
        ),
    )


def _candidate_guard_predictions(
    baseline_predictions: list[str],
    recalls: list[RoutedArticleRecallResult],
    *,
    top_k: int,
) -> list[str]:
    output: list[str] = []
    for baseline, recall in zip(baseline_predictions, recalls):
        names = [candidate.event_name for candidate in recall.candidates[:top_k]]
        if not names or baseline in names:
            output.append(baseline)
        else:
            output.append(names[0])
    return output
