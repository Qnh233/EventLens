from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from eventlens.event_retrieval import EmbeddingClient, EventDefinition, EventSchemaIndex
from eventlens.preprocess import clean_text
from eventlens.schema import ArticleRecord


class SubjectCandidate(BaseModel):
    subject_code: str
    subject_name: str
    score: float = Field(ge=-1.0, le=1.0)


class SubjectRouteResult(BaseModel):
    article_id: str
    scope: str
    accepted_subject_code: str | None = None
    accepted_subject_name: str | None = None
    method: str
    reason: str
    top1_score: float
    top1_margin: float
    candidates: list[SubjectCandidate]


class SubjectRoutingSummary(BaseModel):
    scope: str
    article_count: int
    accepted_count: int
    accepted_rate: float
    exact_alias_count: int
    bge_high_confidence_count: int
    candidate_only_count: int


@dataclass(frozen=True)
class SubjectRoutingPolicy:
    top_k: int = 3
    exact_alias_hard_route: bool = False
    bge_hard_route: bool = False
    score_threshold: float = 1.0
    margin_threshold: float = 1.0


@dataclass(frozen=True)
class _SubjectDefinition:
    code: str
    name: str
    embedding_text: str


class SubjectRouter:
    """事件体系闭集主体召回；高置信才硬路由，其余只返回 Top-K。"""

    def __init__(
        self,
        index: EventSchemaIndex,
        embedding_client: EmbeddingClient,
        *,
        max_query_chars: int = 1600,
        min_alias_chars: int = 2,
    ):
        self.index = index
        self.embedding_client = embedding_client
        self.max_query_chars = max_query_chars
        self.min_alias_chars = min_alias_chars
        self._subject_vectors: dict[str, np.ndarray] = {}

    def route_many(
        self,
        articles: list[ArticleRecord],
        *,
        scope: str,
        policy: SubjectRoutingPolicy,
    ) -> list[SubjectRouteResult]:
        if not articles:
            return []
        query_vectors = np.asarray(
            self.embedding_client.embed([self._query_text(article) for article in articles]),
            dtype=np.float32,
        )
        return self.route_from_vectors(
            articles,
            query_vectors,
            scope=scope,
            policy=policy,
        )

    def route_from_vectors(
        self,
        articles: list[ArticleRecord],
        query_vectors: np.ndarray,
        *,
        scope: str,
        policy: SubjectRoutingPolicy,
    ) -> list[SubjectRouteResult]:
        subjects = self._subjects(scope)
        if not subjects:
            return []
        vectors = np.asarray(query_vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(articles):
            raise ValueError("query_vectors 形状与文章数量不一致")
        subject_vectors = self._subject_embeddings(scope, subjects)
        if vectors.shape[1] != subject_vectors.shape[1]:
            raise ValueError("文章向量与主体向量维度不一致")
        scores = _normalized(vectors) @ _normalized(subject_vectors).T
        return [
            self._route_one(article, row, subjects, policy, scope=scope)
            for article, row in zip(articles, scores)
        ]

    def _route_one(
        self,
        article: ArticleRecord,
        scores: np.ndarray,
        subjects: list[_SubjectDefinition],
        policy: SubjectRoutingPolicy,
        *,
        scope: str,
    ) -> SubjectRouteResult:
        order = np.argsort(-scores)
        top_k = max(1, min(policy.top_k, len(subjects)))
        candidates = [
            SubjectCandidate(
                subject_code=subjects[index].code,
                subject_name=subjects[index].name,
                score=round(float(scores[index]), 6),
            )
            for index in order[:top_k]
        ]
        top1 = int(order[0])
        top1_score = float(scores[top1])
        second_score = float(scores[int(order[1])]) if len(order) > 1 else -1.0
        margin = top1_score - second_score
        exact_hits = self._exact_alias_hits(article, subjects)

        accepted_index: int | None = None
        method = "candidate_only"
        reason = "hard_route_gate_not_met"
        if len(exact_hits) == 1 and policy.exact_alias_hard_route:
            accepted_index = next(iter(exact_hits))
            method = "exact_alias"
            reason = "unique_exact_alias"
        elif (
            not exact_hits
            and policy.bge_hard_route
            and top1_score >= policy.score_threshold
            and margin >= policy.margin_threshold
        ):
            accepted_index = top1
            method = "bge_high_confidence"
            reason = "score_and_margin_gate_met"
        elif len(exact_hits) > 1:
            reason = "ambiguous_exact_alias"
        elif exact_hits:
            reason = "exact_alias_hard_route_disabled"
        elif not policy.bge_hard_route:
            reason = "bge_hard_route_disabled"

        accepted = subjects[accepted_index] if accepted_index is not None else None
        return SubjectRouteResult(
            article_id=article.article_id,
            scope=scope,
            accepted_subject_code=accepted.code if accepted else None,
            accepted_subject_name=accepted.name if accepted else None,
            method=method,
            reason=reason,
            top1_score=round(top1_score, 6),
            top1_margin=round(margin, 6),
            candidates=candidates,
        )

    def _subjects(self, scope: str) -> list[_SubjectDefinition]:
        if scope not in {"company", "industry"}:
            raise ValueError("scope 必须是 company 或 industry")
        by_code = (
            self.index.by_company_code if scope == "company" else self.index.by_industry_code
        )
        label = "公司主体" if scope == "company" else "行业主体"
        output: list[_SubjectDefinition] = []
        for code in sorted(by_code):
            definitions = by_code[code]
            name = definitions[0].subject_name
            event_text = "；".join(
                f"{row.event_name}：{row.description}" for row in definitions
            )
            output.append(
                _SubjectDefinition(
                    code=code,
                    name=name,
                    embedding_text=f"{label}：{name}。事件体系：{event_text}"[:1800],
                )
            )
        return output

    def _subject_embeddings(
        self, scope: str, subjects: list[_SubjectDefinition]
    ) -> np.ndarray:
        if scope not in self._subject_vectors:
            self._subject_vectors[scope] = np.asarray(
                self.embedding_client.embed([row.embedding_text for row in subjects]),
                dtype=np.float32,
            )
        return self._subject_vectors[scope]

    def _query_text(self, article: ArticleRecord) -> str:
        return (
            f"标题：{clean_text(article.title)}\n"
            f"正文：{clean_text(article.content)}"
        )[: self.max_query_chars]

    def _exact_alias_hits(
        self, article: ArticleRecord, subjects: list[_SubjectDefinition]
    ) -> set[int]:
        text = _compact(f"{article.title} {(article.content or '')[:240]}")
        hits: set[int] = set()
        for index, subject in enumerate(subjects):
            aliases = {_compact(subject.name), _compact(subject.code)}
            if any(
                alias and len(alias) >= self.min_alias_chars and alias in text
                for alias in aliases
            ):
                hits.add(index)
        return hits


def _compact(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", clean_text(value).casefold())


def _normalized(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def summarize_subject_routes(
    rows: list[SubjectRouteResult], *, scope: str
) -> SubjectRoutingSummary:
    article_count = len(rows)
    accepted = sum(row.accepted_subject_code is not None for row in rows)
    return SubjectRoutingSummary(
        scope=scope,
        article_count=article_count,
        accepted_count=accepted,
        accepted_rate=round(accepted / max(1, article_count), 6),
        exact_alias_count=sum(row.method == "exact_alias" for row in rows),
        bge_high_confidence_count=sum(
            row.method == "bge_high_confidence" for row in rows
        ),
        candidate_only_count=sum(row.method == "candidate_only" for row in rows),
    )


def load_subject_routes_jsonl(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[SubjectRouteResult]:
    rows: list[SubjectRouteResult] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(SubjectRouteResult.model_validate_json(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows
