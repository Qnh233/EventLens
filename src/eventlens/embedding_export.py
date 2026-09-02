from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from eventlens.event_retrieval import EmbeddingClient
from eventlens.preprocess import clean_text
from eventlens.schema import ArticleRecord


class EmbeddingExportManifest(BaseModel):
    model_id: str
    article_count: int
    completed_count: int
    dimension: int
    dtype: str = "float32"
    max_content_chars: int
    chunk_size: int
    dataset_fingerprint: str
    vectors_path: str = "vectors.npy"
    index_path: str = "index.jsonl"

    @property
    def complete(self) -> bool:
        return self.completed_count == self.article_count


def export_article_embeddings(
    articles: list[ArticleRecord],
    embedding_client: EmbeddingClient,
    *,
    output_dir: str | Path,
    model_id: str,
    max_content_chars: int = 1600,
    chunk_size: int = 1024,
) -> EmbeddingExportManifest:
    """按块写入 float32 .npy；每个块完成后更新 manifest，支持断点续跑。"""

    if not articles:
        raise ValueError("至少需要 1 篇文章")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    vectors_path = output / "vectors.npy"
    index_path = output / "index.jsonl"
    fingerprint = _dataset_fingerprint(articles)
    chunk_size = max(1, chunk_size)

    manifest = _load_manifest(manifest_path)
    if manifest is not None:
        _validate_resume(
            manifest,
            model_id=model_id,
            article_count=len(articles),
            fingerprint=fingerprint,
            max_content_chars=max_content_chars,
        )
        if manifest.complete:
            return manifest
        vectors = np.load(vectors_path, mmap_mode="r+")
        start = manifest.completed_count
    else:
        first_end = min(chunk_size, len(articles))
        first_vectors = _embed_chunk(
            articles[:first_end], embedding_client, max_content_chars=max_content_chars
        )
        dimension = first_vectors.shape[1]
        vectors = np.lib.format.open_memmap(
            vectors_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(articles), dimension),
        )
        vectors[:first_end] = first_vectors
        vectors.flush()
        _write_index(index_path, articles)
        manifest = EmbeddingExportManifest(
            model_id=model_id,
            article_count=len(articles),
            completed_count=first_end,
            dimension=dimension,
            max_content_chars=max_content_chars,
            chunk_size=chunk_size,
            dataset_fingerprint=fingerprint,
        )
        _write_manifest(manifest_path, manifest)
        start = first_end

    for offset in range(start, len(articles), chunk_size):
        end = min(offset + chunk_size, len(articles))
        batch = _embed_chunk(
            articles[offset:end],
            embedding_client,
            max_content_chars=max_content_chars,
        )
        if batch.shape[1] != manifest.dimension:
            raise RuntimeError("embedding 维度发生变化，拒绝继续写入")
        vectors[offset:end] = batch
        vectors.flush()
        manifest = manifest.model_copy(update={"completed_count": end})
        _write_manifest(manifest_path, manifest)
    return manifest


def load_exported_vectors(output_dir: str | Path) -> tuple[EmbeddingExportManifest, np.ndarray]:
    output = Path(output_dir)
    manifest = EmbeddingExportManifest.model_validate_json(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    return manifest, np.load(output / manifest.vectors_path, mmap_mode="r")


def load_exported_article_ids(output_dir: str | Path) -> list[str]:
    """读取向量行对应的 article_id，并校验 index 行号连续。"""

    output = Path(output_dir)
    manifest = EmbeddingExportManifest.model_validate_json(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    article_ids: list[str] = []
    with (output / manifest.index_path).open("r", encoding="utf-8") as file:
        for expected_row, line in enumerate(file):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("row") != expected_row:
                raise ValueError("embedding index 行号不连续")
            article_ids.append(str(payload.get("article_id", "")))
    if len(article_ids) != manifest.article_count:
        raise ValueError("embedding index 数量与 manifest 不一致")
    return article_ids


def article_embedding_text(article: ArticleRecord, *, max_content_chars: int) -> str:
    return (
        f"标题：{clean_text(article.title)}\n"
        f"正文：{clean_text(article.content)[:max_content_chars]}"
    )


def _embed_chunk(
    articles: list[ArticleRecord],
    embedding_client: EmbeddingClient,
    *,
    max_content_chars: int,
) -> np.ndarray:
    rows = embedding_client.embed(
        [
            article_embedding_text(article, max_content_chars=max_content_chars)
            for article in articles
        ]
    )
    vectors = np.asarray(rows, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(articles):
        raise RuntimeError("embedding 返回形状与文章数量不一致")
    return vectors


def _dataset_fingerprint(articles: list[ArticleRecord]) -> str:
    digest = hashlib.sha256()
    for article in articles:
        digest.update(article.article_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_index(path: Path, articles: list[ArticleRecord]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for index, article in enumerate(articles):
            file.write(
                json.dumps(
                    {"row": index, "article_id": article.article_id},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _load_manifest(path: Path) -> EmbeddingExportManifest | None:
    if not path.exists():
        return None
    return EmbeddingExportManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: EmbeddingExportManifest) -> None:
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def _validate_resume(
    manifest: EmbeddingExportManifest,
    *,
    model_id: str,
    article_count: int,
    fingerprint: str,
    max_content_chars: int,
) -> None:
    if manifest.model_id != model_id:
        raise ValueError("model_id 与已有 embedding manifest 不一致")
    if manifest.article_count != article_count or manifest.dataset_fingerprint != fingerprint:
        raise ValueError("输入文章集合与已有 embedding manifest 不一致")
    if manifest.max_content_chars != max_content_chars:
        raise ValueError("文本截断配置与已有 embedding manifest 不一致")
