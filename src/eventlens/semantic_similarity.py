from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Protocol

from eventlens.event_retrieval import EmbeddingClient


class SemanticPairScorer(Protocol):
    def score_pairs(self, pairs: list[tuple[str, str, str]]) -> dict[str, float]: ...


class BgeSemanticPairScorer:
    """批量编码唯一文本，并用 SQLite 复用跨运行 embedding。"""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        *,
        model: str,
        cache_path: str | Path | None = None,
        persist_batch_size: int = 32,
    ):
        self.embedding_client = embedding_client
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self.persist_batch_size = max(1, persist_batch_size)
        self._memory: dict[str, list[float]] = {}

    def score_pairs(self, pairs: list[tuple[str, str, str]]) -> dict[str, float]:
        unique_texts = list(dict.fromkeys(text for _, left, right in pairs for text in (left, right)))
        vectors = self._load_vectors(unique_texts)
        return {
            pair_id: round(_cosine_similarity(vectors[left], vectors[right]), 6)
            for pair_id, left, right in pairs
        }

    def _load_vectors(self, texts: list[str]) -> dict[str, list[float]]:
        missing = [text for text in texts if text not in self._memory]
        if missing and self.cache_path:
            self._memory.update(self._read_cache(missing))
            missing = [text for text in missing if text not in self._memory]
        if missing:
            for start in range(0, len(missing), self.persist_batch_size):
                batch = missing[start : start + self.persist_batch_size]
                embedded = self.embedding_client.embed(batch)
                if len(embedded) != len(batch):
                    raise RuntimeError("embedding 返回数量与输入文本数量不一致")
                fresh = {
                    text: [float(value) for value in vector]
                    for text, vector in zip(batch, embedded)
                }
                self._memory.update(fresh)
                if self.cache_path:
                    self._write_cache(fresh)
        return {text: self._memory[text] for text in texts}

    def _read_cache(self, texts: list[str]) -> dict[str, list[float]]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        hashes = {_text_hash(text): text for text in texts}
        output: dict[str, list[float]] = {}
        with sqlite3.connect(self.cache_path) as connection:
            self._ensure_table(connection)
            for text_hash, text in hashes.items():
                row = connection.execute(
                    "SELECT vector_json FROM embeddings WHERE model = ? AND text_hash = ?",
                    (self.model, text_hash),
                ).fetchone()
                if row:
                    output[text] = [float(value) for value in json.loads(row[0])]
        return output

    def _write_cache(self, vectors: dict[str, list[float]]) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as connection:
            self._ensure_table(connection)
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings(model, text_hash, vector_json) VALUES (?, ?, ?)",
                [
                    (self.model, _text_hash(text), json.dumps(vector, separators=(",", ":")))
                    for text, vector in vectors.items()
                ],
            )

    @staticmethod
    def _ensure_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings(
                model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY(model, text_hash)
            )
            """
        )


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding 维度不一致")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
