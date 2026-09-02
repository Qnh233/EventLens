from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eventlens.preprocess import clean_text
from eventlens.schema import ArticleRecord


@dataclass(frozen=True)
class TaptConfig:
    model: str = "hfl/chinese-macbert-base"
    max_length: int = 256
    max_content_chars: int = 1800
    batch_size: int = 16
    max_steps: int = 1000
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    mlm_probability: float = 0.15
    random_state: int = 42


def build_tapt_text(article: ArticleRecord, *, max_content_chars: int = 1800) -> str:
    """TAPT 只使用生产可见字段，避免把 labeled-only 主体字段泄露进预训练。"""
    parts = [
        f"标题：{clean_text(article.title)}",
        f"来源：{clean_text(article.source)}",
        f"正文：{clean_text(article.content)[:max_content_chars]}",
    ]
    return " ".join(part for part in parts if part.split("：", 1)[-1])


def run_tapt_mlm(
    articles: list[ArticleRecord],
    *,
    config: TaptConfig,
    output_dir: str | Path,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    device: str = "cuda",
) -> dict[str, float | int | str]:
    if not articles:
        raise ValueError("TAPT 至少需要 1 篇无标签文章")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:  # pragma: no cover - 仅远端 GPU 环境执行
        raise RuntimeError("TAPT 需要 torch + transformers") from exc

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    texts = [
        build_tapt_text(article, max_content_chars=config.max_content_chars)
        for article in articles
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        config.model,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    resolved_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model.to(resolved_device)

    class TextDataset(Dataset):
        def __len__(self) -> int:
            return len(texts)

        def __getitem__(self, index: int) -> dict:
            return tokenizer(
                texts[index],
                truncation=True,
                max_length=config.max_length,
                return_special_tokens_mask=True,
            )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
        return_tensors="pt",
    )
    generator = torch.Generator().manual_seed(config.random_state)
    loader = DataLoader(
        TextDataset(),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = min(config.max_steps, max(1, len(loader)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    model.train()
    losses: list[float] = []
    started = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        if step > total_steps:
            break
        inputs = {key: value.to(resolved_device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

    elapsed = time.perf_counter() - started
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(resolved_device) / 1024**2
        if resolved_device.type == "cuda"
        else 0.0
    )
    return {
        "model": config.model,
        "article_count": len(articles),
        "steps": len(losses),
        "mean_mlm_loss": round(float(np.mean(losses)), 6),
        "final_mlm_loss": round(float(losses[-1]), 6),
        "training_seconds": round(elapsed, 3),
        "peak_vram_mb": round(float(peak_vram_mb), 1),
        "output_dir": str(output_path),
    }
