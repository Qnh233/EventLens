from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from eventlens.schema import ArticleRecord


@dataclass(frozen=True)
class EventPair:
    article_index: int
    article_text: str
    event_name: str
    event_text: str
    target: float


def build_article_text(article: ArticleRecord, *, max_content_chars: int = 1800) -> str:
    return f"标题：{article.title}\n来源：{article.source}\n正文：{(article.content or '')[:max_content_chars]}"


def build_event_text(event_name: str, description: str) -> str:
    return f"事件类型：{event_name}\n事件定义：{description}"


def merge_candidate_names(*candidate_groups: Iterable[str], limit: int | None = None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for group in candidate_groups:
        for value in group:
            name = str(value).strip()
            if not name or name in seen:
                continue
            output.append(name)
            seen.add(name)
            if limit is not None and len(output) >= limit:
                return output
    return output


def build_training_pairs(
    articles: list[ArticleRecord],
    truths: list[str],
    candidate_names: list[list[str]],
    descriptions: dict[str, str],
    *,
    indices: list[int],
    max_negatives: int = 5,
    max_content_chars: int = 1800,
) -> list[EventPair]:
    pairs: list[EventPair] = []
    for index in indices:
        truth = truths[index]
        article_text = build_article_text(articles[index], max_content_chars=max_content_chars)
        pairs.append(
            EventPair(
                article_index=index,
                article_text=article_text,
                event_name=truth,
                event_text=build_event_text(truth, descriptions.get(truth, "")),
                target=1.0,
            )
        )
        negatives = [name for name in candidate_names[index] if name != truth][:max_negatives]
        for name in negatives:
            pairs.append(
                EventPair(
                    article_index=index,
                    article_text=article_text,
                    event_name=name,
                    event_text=build_event_text(name, descriptions.get(name, "")),
                    target=0.0,
                )
            )
    return pairs


def choose_ranked_events(
    candidate_names: list[list[str]],
    candidate_scores: list[list[float]],
) -> list[str]:
    if len(candidate_names) != len(candidate_scores):
        raise ValueError("candidate_names 与 candidate_scores 数量不一致")
    output: list[str] = []
    for names, scores in zip(candidate_names, candidate_scores):
        if not names or len(names) != len(scores):
            raise ValueError("每篇文章必须有等长的候选事件与分数")
        best = max(range(len(names)), key=lambda idx: scores[idx])
        output.append(names[best])
    return output
