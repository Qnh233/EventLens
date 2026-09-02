from __future__ import annotations

import hashlib
import heapq
import itertools
import math
import random
import re
from collections import defaultdict
from datetime import datetime

from pydantic import BaseModel, Field

from eventlens.preprocess import clean_text
from eventlens.schema import ArticleRecord


class DuplicatePair(BaseModel):
    pair_id: str
    scope: str
    label: int = Field(ge=0, le=1)
    left_article_id: str
    right_article_id: str
    left_duplication_id: str
    right_duplication_id: str
    subject_key: str
    subject_name: str = ""
    subject_resolution_method: str = "unresolved"
    subject_resolution_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    event_label: str | None = None
    time_gap_days: float | None = None
    title_similarity: float = Field(ge=0.0, le=1.0)
    left_text: str
    right_text: str


class DuplicatePairSummary(BaseModel):
    pair_count: int
    positive_count: int
    negative_count: int
    subject_count: int
    positive_group_count: int
    resolved_pair_count: int = 0
    positive_resolved_count: int = 0
    negative_resolved_count: int = 0
    negative_subject_count: int = 0
    resolution_method_distribution: dict[str, int] = Field(default_factory=dict)


class DuplicateClusterGroup(BaseModel):
    duplication_id: str
    subject_key: str
    subject_name: str = ""
    event_label: str = "同源事件"
    subject_resolution_method: str
    subject_resolution_confidence: float = Field(ge=0.0, le=1.0)
    articles: list[ArticleRecord]


class _GroupContext(BaseModel):
    duplication_id: str
    subject_key: str
    subject_name: str = ""
    subject_resolution_method: str = "unresolved"
    subject_resolution_confidence: float = 0.0
    event_label: str | None = None
    articles: list[ArticleRecord]
    anchor_time: datetime | None = None


def build_duplicate_pairs(
    datasets: dict[str, list[ArticleRecord]],
    *,
    scope: str,
    negative_ratio: float = 1.0,
    max_positive_pairs_per_group: int = 20,
    time_window_days: int = 7,
    max_text_chars: int = 1200,
    seed: int = 42,
    max_pairs: int | None = None,
    subject_lead_chars: int = 240,
    min_subject_alias_chars: int = 2,
    require_resolved_subject_for_negatives: bool = True,
) -> list[DuplicatePair]:
    """用 duplication_id 构造正样本，并从同主体近时间不同组中挖掘难负样本。"""

    if scope not in {"company", "industry"}:
        raise ValueError("scope 必须是 company 或 industry")
    event_key = f"{scope}_event"
    duplicate_key = f"{scope}_duplicate"
    event_articles = datasets.get(event_key, [])
    contexts = _build_group_contexts(
        [*event_articles, *datasets.get(duplicate_key, [])],
        scope=scope,
        subject_catalog=_build_subject_catalog(event_articles, scope=scope),
        subject_lead_chars=subject_lead_chars,
        min_subject_alias_chars=min_subject_alias_chars,
    )

    positives: list[DuplicatePair] = []
    for context in contexts:
        pairs = list(itertools.combinations(context.articles, 2))
        if len(pairs) > max_positive_pairs_per_group:
            rng = random.Random(f"{seed}:{context.duplication_id}")
            rng.shuffle(pairs)
            pairs = pairs[:max_positive_pairs_per_group]
        positives.extend(
            _to_pair(
                left,
                right,
                left_context=context,
                right_context=context,
                scope=scope,
                label=1,
                max_text_chars=max_text_chars,
            )
            for left, right in pairs
        )
    positives.sort(key=lambda pair: pair.pair_id)

    if max_pairs is not None:
        ratio = max(0.0, negative_ratio)
        positive_limit = max_pairs if ratio == 0 else max(1, int(max_pairs / (1 + ratio)))
        positives = positives[:positive_limit]

    target_negative_count = math.ceil(len(positives) * max(0.0, negative_ratio))
    if max_pairs is not None:
        target_negative_count = min(target_negative_count, max_pairs - len(positives))
    negatives = _hard_negative_pairs(
        contexts,
        scope=scope,
        target_count=target_negative_count,
        time_window_days=time_window_days,
        max_text_chars=max_text_chars,
        require_resolved_subject=require_resolved_subject_for_negatives,
    )
    return [*positives, *negatives]


def summarize_duplicate_pairs(pairs: list[DuplicatePair]) -> DuplicatePairSummary:
    positive_groups = {
        pair.left_duplication_id for pair in pairs if pair.label == 1
    }
    methods: dict[str, int] = defaultdict(int)
    for pair in pairs:
        methods[pair.subject_resolution_method] += 1
    return DuplicatePairSummary(
        pair_count=len(pairs),
        positive_count=sum(pair.label == 1 for pair in pairs),
        negative_count=sum(pair.label == 0 for pair in pairs),
        subject_count=len({pair.subject_key for pair in pairs if pair.subject_key}),
        positive_group_count=len(positive_groups),
        resolved_pair_count=sum(bool(pair.subject_key) for pair in pairs),
        positive_resolved_count=sum(
            pair.label == 1 and bool(pair.subject_key) for pair in pairs
        ),
        negative_resolved_count=sum(
            pair.label == 0 and bool(pair.subject_key) for pair in pairs
        ),
        negative_subject_count=len(
            {pair.subject_key for pair in pairs if pair.label == 0 and pair.subject_key}
        ),
        resolution_method_distribution=dict(sorted(methods.items())),
    )


def build_duplicate_cluster_groups(
    datasets: dict[str, list[ArticleRecord]],
    *,
    scope: str,
    max_articles: int | None = None,
    subject_lead_chars: int = 240,
    min_subject_alias_chars: int = 2,
    require_multiple_groups_per_subject: bool = True,
) -> list[DuplicateClusterGroup]:
    """构造无歧义的完整真值簇，用于事件簇级评测。"""

    if scope not in {"company", "industry"}:
        raise ValueError("scope 必须是 company 或 industry")
    event_articles = datasets.get(f"{scope}_event", [])
    duplicate_articles = datasets.get(f"{scope}_duplicate", [])
    contexts = _build_group_contexts(
        [*event_articles, *duplicate_articles],
        scope=scope,
        subject_catalog=_build_subject_catalog(event_articles, scope=scope),
        subject_lead_chars=subject_lead_chars,
        min_subject_alias_chars=min_subject_alias_chars,
    )

    identity_counts: dict[tuple[str, str], int] = defaultdict(int)
    for context in contexts:
        for article in context.articles:
            identity_counts[(article.article_id, _content_fingerprint(article))] += 1

    groups: list[DuplicateClusterGroup] = []
    for context in contexts:
        if not context.subject_key:
            continue
        unambiguous = [
            article
            for article in context.articles
            if identity_counts[(article.article_id, _content_fingerprint(article))] == 1
        ]
        if len(unambiguous) < 2:
            continue
        articles = [
            _benchmark_article(
                article,
                context=context,
                scope=scope,
                index=index,
            )
            for index, article in enumerate(unambiguous, start=1)
        ]
        groups.append(
            DuplicateClusterGroup(
                duplication_id=context.duplication_id,
                subject_key=context.subject_key,
                subject_name=context.subject_name,
                event_label=context.event_label or "同源事件",
                subject_resolution_method=context.subject_resolution_method,
                subject_resolution_confidence=context.subject_resolution_confidence,
                articles=articles,
            )
        )

    by_subject: dict[str, list[DuplicateClusterGroup]] = defaultdict(list)
    for group in groups:
        by_subject[group.subject_key].append(group)
    if require_multiple_groups_per_subject:
        by_subject = {
            subject: rows for subject, rows in by_subject.items() if len(rows) >= 2
        }

    for rows in by_subject.values():
        rows.sort(key=lambda row: (row.duplication_id, len(row.articles)))
    ordered_subjects = sorted(
        by_subject,
        key=lambda subject: (-len(by_subject[subject]), subject),
    )
    selected: list[DuplicateClusterGroup] = []
    article_count = 0
    cursor = 0
    while ordered_subjects:
        progressed = False
        for subject in list(ordered_subjects):
            rows = by_subject[subject]
            if cursor >= len(rows):
                continue
            group = rows[cursor]
            if max_articles is not None and article_count + len(group.articles) > max_articles:
                continue
            selected.append(group)
            article_count += len(group.articles)
            progressed = True
        cursor += 1
        ordered_subjects = [
            subject for subject in ordered_subjects if cursor < len(by_subject[subject])
        ]
        if not progressed:
            break
    return selected


def _build_group_contexts(
    articles: list[ArticleRecord],
    *,
    scope: str,
    subject_catalog: dict[str, str],
    subject_lead_chars: int,
    min_subject_alias_chars: int,
) -> list[_GroupContext]:
    grouped: dict[str, list[ArticleRecord]] = defaultdict(list)
    for article in articles:
        for duplication_id in _split_duplication_ids(article.duplication_id):
            grouped[duplication_id].append(article)

    contexts: list[_GroupContext] = []
    for duplication_id, rows in sorted(grouped.items()):
        unique_rows = _deduplicate_rows(rows)
        if len(unique_rows) < 2:
            continue
        subject_key, subject_name, method, confidence = _resolve_group_subject(
            unique_rows,
            scope=scope,
            subject_catalog=subject_catalog,
            subject_lead_chars=subject_lead_chars,
            min_subject_alias_chars=min_subject_alias_chars,
        )
        dated = [row.publish_time for row in unique_rows if row.publish_time]
        contexts.append(
            _GroupContext(
                duplication_id=duplication_id,
                subject_key=subject_key,
                subject_name=subject_name,
                subject_resolution_method=method,
                subject_resolution_confidence=confidence,
                event_label=next(
                    (row.event_label for row in unique_rows if row.event_label),
                    None,
                ),
                articles=unique_rows,
                anchor_time=min(dated) if dated else None,
            )
        )
    return contexts


def _hard_negative_pairs(
    contexts: list[_GroupContext],
    *,
    scope: str,
    target_count: int,
    time_window_days: int,
    max_text_chars: int,
    require_resolved_subject: bool,
) -> list[DuplicatePair]:
    if target_count <= 0:
        return []
    by_subject: dict[str, list[_GroupContext]] = defaultdict(list)
    for context in contexts:
        if require_resolved_subject and not context.subject_key:
            continue
        bucket = context.subject_key or "__UNKNOWN_SUBJECT__"
        by_subject[bucket].append(context)

    heap: list[
        tuple[
            float,
            float,
            str,
            ArticleRecord,
            ArticleRecord,
            _GroupContext,
            _GroupContext,
        ]
    ] = []
    seen_article_pairs: set[tuple[str, str]] = set()
    for subject_contexts in by_subject.values():
        for left_context, right_context in itertools.combinations(subject_contexts, 2):
            group_gap = _time_gap_days(left_context.anchor_time, right_context.anchor_time)
            if group_gap is not None and group_gap > time_window_days:
                continue
            for left in left_context.articles:
                for right in right_context.articles:
                    article_key = tuple(sorted((left.article_id, right.article_id)))
                    if article_key in seen_article_pairs:
                        continue
                    if _content_fingerprint(left) == _content_fingerprint(right):
                        continue
                    seen_article_pairs.add(article_key)
                    similarity = _title_similarity(left.title, right.title)
                    gap = _time_gap_days(left.publish_time, right.publish_time)
                    gap_rank = gap if gap is not None else 10**9
                    pair_id = _pair_id(
                        left,
                        right,
                        left_context=left_context,
                        right_context=right_context,
                        scope=scope,
                        label=0,
                    )
                    entry = (
                        similarity,
                        -gap_rank,
                        pair_id,
                        left,
                        right,
                        left_context,
                        right_context,
                    )
                    if len(heap) < target_count:
                        heapq.heappush(heap, entry)
                    elif entry[:3] > heap[0][:3]:
                        heapq.heapreplace(heap, entry)

    candidates = [
        _to_pair(
            left,
            right,
            left_context=left_context,
            right_context=right_context,
            scope=scope,
            label=0,
            max_text_chars=max_text_chars,
        )
        for _, _, _, left, right, left_context, right_context in heap
    ]
    candidates.sort(
        key=lambda pair: (
            -pair.title_similarity,
            pair.time_gap_days if pair.time_gap_days is not None else 10**9,
            pair.pair_id,
        )
    )
    return candidates[:target_count]


def _to_pair(
    left: ArticleRecord,
    right: ArticleRecord,
    *,
    left_context: _GroupContext,
    right_context: _GroupContext,
    scope: str,
    label: int,
    max_text_chars: int,
) -> DuplicatePair:
    left_article, right_article = sorted(
        (left, right), key=lambda article: article.article_id
    )
    left_group = (
        left_context if left_article.article_id == left.article_id else right_context
    )
    right_group = (
        right_context if right_article.article_id == right.article_id else left_context
    )
    pair_id = _pair_id(
        left_article,
        right_article,
        left_context=left_group,
        right_context=right_group,
        scope=scope,
        label=label,
    )
    return DuplicatePair(
        pair_id=pair_id,
        scope=scope,
        label=label,
        left_article_id=left_article.article_id,
        right_article_id=right_article.article_id,
        left_duplication_id=left_group.duplication_id,
        right_duplication_id=right_group.duplication_id,
        subject_key=left_context.subject_key,
        subject_name=left_context.subject_name,
        subject_resolution_method=_pair_resolution_method(left_context, right_context),
        subject_resolution_confidence=min(
            left_context.subject_resolution_confidence,
            right_context.subject_resolution_confidence,
        ),
        event_label=left_context.event_label
        if left_context.event_label == right_context.event_label
        else None,
        time_gap_days=_time_gap_days(
            left_article.publish_time, right_article.publish_time
        ),
        title_similarity=_title_similarity(left_article.title, right_article.title),
        left_text=_article_text(left_article, max_text_chars),
        right_text=_article_text(right_article, max_text_chars),
    )


def _pair_id(
    left: ArticleRecord,
    right: ArticleRecord,
    *,
    left_context: _GroupContext,
    right_context: _GroupContext,
    scope: str,
    label: int,
) -> str:
    left_article, right_article = sorted(
        (left, right), key=lambda article: article.article_id
    )
    left_group = (
        left_context if left_article.article_id == left.article_id else right_context
    )
    right_group = (
        right_context if right_article.article_id == right.article_id else left_context
    )
    seed = (
        f"{scope}|{label}|{left_article.article_id}|{right_article.article_id}|"
        f"{left_group.duplication_id}|{right_group.duplication_id}"
    )
    return f"PAIR-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _subject_key(article: ArticleRecord, scope: str) -> str:
    return article.trading_code if scope == "company" else article.industry_code


def _subject_name(article: ArticleRecord, scope: str) -> str:
    return article.entity if scope == "company" else article.industry


def _build_subject_catalog(
    articles: list[ArticleRecord], *, scope: str
) -> dict[str, str]:
    catalog: dict[str, str] = {}
    for article in articles:
        key = _subject_key(article, scope)
        name = _subject_name(article, scope)
        if key and name:
            catalog[key] = name
    return catalog


def _resolve_group_subject(
    articles: list[ArticleRecord],
    *,
    scope: str,
    subject_catalog: dict[str, str],
    subject_lead_chars: int,
    min_subject_alias_chars: int,
) -> tuple[str, str, str, float]:
    direct = {
        (_subject_key(article, scope), _subject_name(article, scope))
        for article in articles
        if _subject_key(article, scope)
    }
    if len(direct) == 1:
        key, name = next(iter(direct))
        return key, name or subject_catalog.get(key, ""), "source_field", 1.0
    if len(direct) > 1:
        return "", "", "ambiguous_source_field", 0.0

    title_text = _compact_match_text(" ".join(article.title for article in articles))
    title_hits = _unique_alias_hits(
        title_text,
        subject_catalog,
        min_alias_chars=min_subject_alias_chars,
    )
    if len(title_hits) == 1:
        key = next(iter(title_hits))
        return key, subject_catalog[key], "exact_title_alias", 0.95
    if len(title_hits) > 1:
        return "", "", "ambiguous_title_alias", 0.0

    lead_text = _compact_match_text(
        " ".join(article.content[:subject_lead_chars] for article in articles)
    )
    lead_hits = _unique_alias_hits(
        lead_text,
        subject_catalog,
        min_alias_chars=min_subject_alias_chars,
    )
    if len(lead_hits) == 1:
        key = next(iter(lead_hits))
        return key, subject_catalog[key], "exact_lead_alias", 0.85
    if len(lead_hits) > 1:
        return "", "", "ambiguous_lead_alias", 0.0
    return "", "", "unresolved", 0.0


def _unique_alias_hits(
    text: str,
    subject_catalog: dict[str, str],
    *,
    min_alias_chars: int,
) -> set[str]:
    hits: set[str] = set()
    if not text:
        return hits
    for key, name in subject_catalog.items():
        aliases = {_compact_match_text(name), _compact_match_text(key)}
        if any(
            alias and len(alias) >= min_alias_chars and alias in text
            for alias in aliases
        ):
            hits.add(key)
    return hits


def _compact_match_text(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", clean_text(text).casefold())


def _pair_resolution_method(
    left_context: _GroupContext, right_context: _GroupContext
) -> str:
    if left_context.subject_resolution_method == right_context.subject_resolution_method:
        return left_context.subject_resolution_method
    return "+".join(
        sorted(
            {
                left_context.subject_resolution_method,
                right_context.subject_resolution_method,
            }
        )
    )


def _benchmark_article(
    article: ArticleRecord,
    *,
    context: _GroupContext,
    scope: str,
    index: int,
) -> ArticleRecord:
    seed = (
        f"{context.duplication_id}|{article.article_id}|"
        f"{_content_fingerprint(article)}|{index}"
    )
    article_id = f"BENCH-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"
    updates: dict = {
        "article_id": article_id,
        "duplication_id": context.duplication_id,
        "event_label": context.event_label or "同源事件",
        "task_scope": f"{scope}_event",
        "extra": {
            **article.extra,
            "benchmark_original_article_id": article.article_id,
            "benchmark_subject_resolution_method": context.subject_resolution_method,
        },
    }
    if scope == "company":
        updates.update(
            trading_code=context.subject_key,
            entity=context.subject_name,
        )
    else:
        updates.update(
            industry_code=context.subject_key,
            industry=context.subject_name,
        )
    return article.model_copy(update=updates)


def _deduplicate_rows(rows: list[ArticleRecord]) -> list[ArticleRecord]:
    output: list[ArticleRecord] = []
    seen: set[tuple[str, str]] = set()
    for row in sorted(rows, key=lambda article: (article.article_id, article.source_row or 0)):
        key = (row.article_id, _content_fingerprint(row))
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _content_fingerprint(article: ArticleRecord) -> str:
    text = f"{clean_text(article.title)}\n{clean_text(article.content)}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_duplication_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _article_text(article: ArticleRecord, max_text_chars: int) -> str:
    text = f"{clean_text(article.title)}\n{clean_text(article.content)}".strip()
    return text[:max_text_chars]


def _title_similarity(left: str, right: str) -> float:
    left_tokens = _char_ngrams(clean_text(left), 2)
    right_tokens = _char_ngrams(clean_text(right), 2)
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 6)


def _char_ngrams(text: str, size: int) -> set[str]:
    compact = "".join(text.split())
    if len(compact) < size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _time_gap_days(left: datetime | None, right: datetime | None) -> float | None:
    if left is None or right is None:
        return None
    return round(abs((left - right).total_seconds()) / 86400, 4)
