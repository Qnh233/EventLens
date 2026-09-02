from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from eventlens.preprocess import clean_text
from eventlens.schema import ArticleRecord


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class NativeSentenceTransformerEmbeddingClient:
    """可选原生 GPU provider；依赖仅在实例化时加载。"""

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        cache_folder: str | Path | None = None,
        local_files_only: bool = False,
        encoder=None,
    ):
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "原生 embedding provider 需要安装 sentence-transformers"
                ) from exc
            encoder = SentenceTransformer(
                model,
                device=device,
                cache_folder=str(cache_folder) if cache_folder else None,
                local_files_only=local_files_only,
            )
        self.encoder = encoder
        self.model = model
        self.batch_size = max(1, batch_size)
        self.normalize_embeddings = normalize_embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(value) for value in vector] for vector in vectors]


class EventDefinition(BaseModel):
    scope: str
    subject_code: str
    subject_name: str
    event_name: str
    description: str

    @property
    def definition_id(self) -> str:
        return f"{self.scope}:{self.subject_code}:{self.event_name}"

    def as_embedding_text(self) -> str:
        return (
            f"主体：{self.subject_name}；事件：{self.event_name}；"
            f"事件定义：{self.description}"
        )


class EventCandidate(BaseModel):
    event_name: str
    score: float = Field(ge=-1.0, le=1.0)
    subject_code: str
    subject_name: str
    description: str


class ArticleRecallResult(BaseModel):
    article_id: str
    scope: str
    subject_code: str
    expected_event: str | None = None
    candidate_count: int
    candidates: list[EventCandidate]


class RoutedArticleRecallResult(BaseModel):
    article_id: str
    scope: str
    subject_codes: list[str]
    candidate_count: int
    candidates: list[EventCandidate]


class RecallEvaluation(BaseModel):
    sample_count: int
    schema_covered_count: int
    multi_candidate_count: int
    average_candidate_count: float
    hit_at_1: float
    hit_at_k: float
    mean_reciprocal_rank: float
    top_k: int


class EventSchemaIndex:
    def __init__(self, definitions: list[EventDefinition]):
        self.definitions = definitions
        self.by_company_code: dict[str, list[EventDefinition]] = {}
        self.by_company_name: dict[str, list[EventDefinition]] = {}
        self.by_industry_code: dict[str, list[EventDefinition]] = {}
        self.by_industry_name: dict[str, list[EventDefinition]] = {}
        for definition in definitions:
            if definition.scope == "company":
                self.by_company_code.setdefault(definition.subject_code, []).append(definition)
                self.by_company_name.setdefault(definition.subject_name, []).append(definition)
            else:
                self.by_industry_code.setdefault(definition.subject_code, []).append(definition)
                self.by_industry_name.setdefault(definition.subject_name, []).append(definition)

    @classmethod
    def from_files(
        cls,
        *,
        company_path: str | Path | None = None,
        industry_path: str | Path | None = None,
    ) -> "EventSchemaIndex":
        definitions: list[EventDefinition] = []
        if company_path is not None:
            payload = json.loads(Path(company_path).read_text(encoding="utf-8-sig"))
            for row in payload:
                company = row.get("company", {})
                code = _normalize_code(company.get("trading_code"), width=6)
                name = clean_text(company.get("secu_abbr"))
                definitions.extend(
                    EventDefinition(
                        scope="company",
                        subject_code=code,
                        subject_name=name,
                        event_name=clean_text(event.get("event_name")),
                        description=clean_text(event.get("description")),
                    )
                    for event in row.get("event_schema", [])
                    if clean_text(event.get("event_name"))
                )
        if industry_path is not None:
            payload = json.loads(Path(industry_path).read_text(encoding="utf-8-sig"))
            for row in payload:
                industry = row.get("industry", {})
                code = _normalize_code(industry.get("inducode"))
                name = clean_text(industry.get("induname"))
                definitions.extend(
                    EventDefinition(
                        scope="industry",
                        subject_code=code,
                        subject_name=name,
                        event_name=clean_text(event.get("event_name")),
                        description=clean_text(event.get("description")),
                    )
                    for event in row.get("event_schema", [])
                    if clean_text(event.get("event_name"))
                )
        return cls(definitions)

    def candidates_for(self, article: ArticleRecord) -> list[EventDefinition]:
        if article.task_scope.startswith("industry"):
            return self.by_industry_code.get(article.industry_code) or self.by_industry_name.get(
                article.industry, []
            )
        return self.by_company_code.get(article.trading_code) or self.by_company_name.get(
            article.entity, []
        )


class OllamaEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "bge-m3:latest",
        timeout_seconds: float = 60.0,
        batch_size: int = 8,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
        num_gpu: int | None = None,
    ):
        self.endpoint = f"{base_url.rstrip('/')}/api/embed"
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, max_retries)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self.num_gpu = num_gpu

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            output.extend(self._embed_resilient(texts[start : start + self.batch_size]))
        return output

    def _embed_resilient(
        self, texts: list[str], *, retries_remaining: int | None = None
    ) -> list[list[float]]:
        retries_remaining = (
            self.max_retries if retries_remaining is None else retries_remaining
        )
        try:
            return self._embed_batch(texts)
        except RuntimeError as exc:
            if "cudaMalloc" in str(exc) or "out of memory" in str(exc).lower():
                raise
            if len(texts) <= 1:
                if retries_remaining <= 0:
                    raise
                if self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
                return self._embed_resilient(
                    texts, retries_remaining=retries_remaining - 1
                )
            midpoint = len(texts) // 2
            return [
                *self._embed_resilient(
                    texts[:midpoint], retries_remaining=retries_remaining
                ),
                *self._embed_resilient(
                    texts[midpoint:], retries_remaining=retries_remaining
                ),
            ]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        request_payload: dict = {"model": self.model, "input": texts}
        if self.num_gpu is not None:
            request_payload["options"] = {"num_gpu": self.num_gpu}
        payload = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"Ollama embedding 调用失败：HTTP {exc.code}，{detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Ollama embedding 调用失败：{exc}") from exc
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama embedding 返回数量与输入不一致")
        return [[float(value) for value in vector] for vector in embeddings]


class SubjectConstrainedEventRetriever:
    def __init__(
        self,
        index: EventSchemaIndex,
        embedding_client: EmbeddingClient,
        *,
        max_query_chars: int = 1600,
    ):
        self.index = index
        self.embedding_client = embedding_client
        self.max_query_chars = max_query_chars
        self._definition_embeddings: dict[str, list[float]] = {}

    def recall(self, article: ArticleRecord, top_k: int = 3) -> list[EventCandidate]:
        return self.recall_many([article], top_k=top_k)[0].candidates

    def recall_many(
        self,
        articles: list[ArticleRecord],
        *,
        top_k: int = 3,
    ) -> list[ArticleRecallResult]:
        definitions_by_article = [self.index.candidates_for(article) for article in articles]
        unique_missing: dict[str, EventDefinition] = {}
        for definitions in definitions_by_article:
            for definition in definitions:
                if definition.definition_id not in self._definition_embeddings:
                    unique_missing[definition.definition_id] = definition
        missing = list(unique_missing.values())
        missing = [
            definition
            for definition in missing
            if definition.definition_id not in self._definition_embeddings
        ]
        if missing:
            vectors = self.embedding_client.embed(
                [definition.as_embedding_text() for definition in missing]
            )
            for definition, vector in zip(missing, vectors):
                self._definition_embeddings[definition.definition_id] = vector

        query_vectors = self.embedding_client.embed(
            [self._query_text(article) for article in articles]
        )
        output: list[ArticleRecallResult] = []
        for article, definitions, query_vector in zip(
            articles, definitions_by_article, query_vectors
        ):
            ranked = sorted(
                (
                    EventCandidate(
                        event_name=definition.event_name,
                        score=round(
                            _cosine_similarity(
                                query_vector,
                                self._definition_embeddings[definition.definition_id],
                            ),
                            6,
                        ),
                        subject_code=definition.subject_code,
                        subject_name=definition.subject_name,
                        description=definition.description,
                    )
                    for definition in definitions
                ),
                key=lambda candidate: (-candidate.score, candidate.event_name),
            )
            output.append(
                ArticleRecallResult(
                    article_id=article.article_id,
                    scope="industry"
                    if article.task_scope.startswith("industry")
                    else "company",
                    subject_code=article.industry_code
                    if article.task_scope.startswith("industry")
                    else article.trading_code,
                    expected_event=article.event_label,
                    candidate_count=len(definitions),
                    candidates=ranked[: max(1, top_k)],
                )
            )
        return output

    def evaluate(
        self,
        articles: list[ArticleRecord],
        *,
        top_k: int = 3,
    ) -> RecallEvaluation:
        results = self.recall_many(articles, top_k=top_k)
        return evaluate_recall_results(results, top_k=top_k)

    def recall_from_vectors(
        self,
        article_ids: list[str],
        query_vectors,
        subject_codes_by_article: list[list[str]],
        *,
        scope: str,
        top_k: int = 3,
    ) -> list[RoutedArticleRecallResult]:
        """复用已导出的文章向量，只编码少量事件定义。"""

        if scope not in {"company", "industry"}:
            raise ValueError("scope 必须是 company 或 industry")
        if not (
            len(article_ids) == len(query_vectors) == len(subject_codes_by_article)
        ):
            raise ValueError("文章、向量和主体候选数量不一致")
        by_code = (
            self.index.by_company_code
            if scope == "company"
            else self.index.by_industry_code
        )
        definitions_by_article: list[list[EventDefinition]] = []
        missing: dict[str, EventDefinition] = {}
        for subject_codes in subject_codes_by_article:
            definitions = [
                definition
                for code in subject_codes
                for definition in by_code.get(code, [])
            ]
            definitions_by_article.append(definitions)
            for definition in definitions:
                if definition.definition_id not in self._definition_embeddings:
                    missing[definition.definition_id] = definition
        if missing:
            rows = list(missing.values())
            vectors = self.embedding_client.embed(
                [definition.as_embedding_text() for definition in rows]
            )
            for definition, vector in zip(rows, vectors):
                self._definition_embeddings[definition.definition_id] = vector

        output: list[RoutedArticleRecallResult] = []
        for article_id, query_vector, subject_codes, definitions in zip(
            article_ids,
            query_vectors,
            subject_codes_by_article,
            definitions_by_article,
        ):
            definition_vectors = [
                self._definition_embeddings[definition.definition_id]
                for definition in definitions
            ]
            scores = _cosine_similarities(query_vector, definition_vectors)
            ranked = sorted(
                [
                    EventCandidate(
                        event_name=definition.event_name,
                        score=round(score, 6),
                        subject_code=definition.subject_code,
                        subject_name=definition.subject_name,
                        description=definition.description,
                    )
                    for definition, score in zip(definitions, scores)
                ],
                key=lambda candidate: (-candidate.score, candidate.event_name),
            )
            output.append(
                RoutedArticleRecallResult(
                    article_id=article_id,
                    scope=scope,
                    subject_codes=subject_codes,
                    candidate_count=len(definitions),
                    candidates=ranked[: max(1, top_k)],
                )
            )
        return output

    def _query_text(self, article: ArticleRecord) -> str:
        subject = article.entity or article.industry
        text = (
            f"主体：{subject}；标题：{clean_text(article.title)}；"
            f"正文：{clean_text(article.content)}"
        )
        return text[: self.max_query_chars]


def evaluate_recall_results(
    results: list[ArticleRecallResult],
    *,
    top_k: int,
) -> RecallEvaluation:
    covered = 0
    multi_candidate_count = 0
    candidate_count_total = 0
    hit_1 = 0
    hit_k = 0
    reciprocal_rank = 0.0
    labeled = [result for result in results if result.expected_event]
    for result in labeled:
        if not result.candidates:
            continue
        covered += 1
        candidate_count_total += result.candidate_count
        multi_candidate_count += int(result.candidate_count > 1)
        names = [candidate.event_name for candidate in result.candidates]
        if names and names[0] == result.expected_event:
            hit_1 += 1
        if result.expected_event in names:
            rank = names.index(result.expected_event) + 1
            hit_k += 1
            reciprocal_rank += 1 / rank
    denominator = covered or 1
    return RecallEvaluation(
        sample_count=len(labeled),
        schema_covered_count=covered,
        multi_candidate_count=multi_candidate_count,
        average_candidate_count=round(candidate_count_total / denominator, 4),
        hit_at_1=round(hit_1 / denominator, 6),
        hit_at_k=round(hit_k / denominator, 6),
        mean_reciprocal_rank=round(reciprocal_rank / denominator, 6),
        top_k=top_k,
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding 维度不一致")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _cosine_similarities(query_vector, candidate_vectors: list[list[float]]) -> list[float]:
    """批量计算同一文章与少量事件定义的余弦相似度。"""

    if not candidate_vectors:
        return []
    query = np.asarray(query_vector, dtype=np.float64)
    candidates = np.asarray(candidate_vectors, dtype=np.float64)
    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    denominators = candidate_norms * query_norm
    dots = candidates @ query
    return np.divide(
        dots,
        denominators,
        out=np.zeros_like(dots, dtype=np.float64),
        where=denominators > 0,
    ).tolist()


def _normalize_code(value: object, width: int | None = None) -> str:
    text = clean_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if width and text.isdigit():
        text = text.zfill(width)
    return text
