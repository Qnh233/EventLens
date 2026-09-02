from __future__ import annotations

from collections import Counter
import math
import random
import time
from dataclasses import dataclass

import numpy as np

from eventlens.label_anchor_contrastive import (
    build_production_text,
    freeze_except_last_layers,
    normalized_cls,
)
from eventlens.schema import ArticleRecord


@dataclass(frozen=True)
class ArticleTripletTrainingConfig:
    model: str
    max_length: int = 384
    max_content_chars: int = 2400
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    margin: float = 0.08
    trainable_last_layers: int = 2
    random_state: int = 42
    class_balanced_sampling: bool = False


def regularized_class_balance_anchor_weights(labels: list[str]) -> list[float]:
    """按类频次平方根倒数加权，缓和长尾过采样带来的训练方差。"""

    counts = Counter(labels)
    if not labels or any(count <= 0 for count in counts.values()):
        raise ValueError("labels must be non-empty")
    return [1.0 / math.sqrt(counts[label]) for label in labels]


def build_triplet_indices(
    labels: list[str],
    classes: list[str],
    hard_negative_ids_by_class: dict[int, list[int]],
    *,
    random_state: int,
) -> list[tuple[int, int, int]]:
    """为每个 Gold 样本构造同类正样本和自动混淆负样本。"""

    class_to_id = {label: index for index, label in enumerate(classes)}
    indices_by_class: dict[int, list[int]] = {index: [] for index in range(len(classes))}
    for index, label in enumerate(labels):
        if label not in class_to_id:
            raise ValueError(f"unknown label: {label}")
        indices_by_class[class_to_id[label]].append(index)
    if any(len(values) < 2 for values in indices_by_class.values()):
        raise ValueError("triplet training requires at least two samples per class")

    rng = random.Random(random_state)
    triplets: list[tuple[int, int, int]] = []
    for anchor_index, label in enumerate(labels):
        class_id = class_to_id[label]
        positives = [index for index in indices_by_class[class_id] if index != anchor_index]
        positive_index = rng.choice(positives)

        negative_classes = [
            int(value)
            for value in hard_negative_ids_by_class.get(class_id, [])
            if int(value) != class_id and indices_by_class.get(int(value))
        ]
        if not negative_classes:
            negative_classes = [
                index for index in range(len(classes)) if index != class_id and indices_by_class[index]
            ]
        negative_class = rng.choice(negative_classes)
        negative_index = rng.choice(indices_by_class[negative_class])
        triplets.append((anchor_index, positive_index, negative_index))
    return triplets


def train_article_triplet_encoder(
    articles: list[ArticleRecord],
    labels: list[str],
    *,
    classes: list[str],
    hard_negative_ids_by_class: dict[int, list[int]],
    config: ArticleTripletTrainingConfig,
    local_files_only: bool = True,
    device: str = "cuda",
) -> dict:
    """只微调 BGE 最后少量层，以真实文章 triplet 学习细粒度事件边界。"""

    if len(articles) != len(labels):
        raise ValueError("article/label count mismatch")
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("article contrastive training requires torch + transformers") from exc

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(config.model, local_files_only=local_files_only).to(resolved_device)
    trainable_parameters, total_parameters = freeze_except_last_layers(
        model, last_n=config.trainable_last_layers
    )
    texts = [
        build_production_text(article, max_content_chars=config.max_content_chars)
        for article in articles
    ]
    triplets = build_triplet_indices(
        labels,
        classes,
        hard_negative_ids_by_class,
        random_state=config.random_state,
    )

    class TripletDataset(Dataset):
        def __len__(self) -> int:
            return len(triplets)

        def __getitem__(self, index: int):
            anchor, positive, negative = triplets[index]
            return texts[anchor], texts[positive], texts[negative]

    def collate(rows):
        flat: list[str] = []
        for anchor, positive, negative in rows:
            flat.extend([anchor, positive, negative])
        return tokenizer(
            flat,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )

    loader_generator = torch.Generator().manual_seed(config.random_state)
    sampler = None
    if config.class_balanced_sampling:
        sampler = WeightedRandomSampler(
            regularized_class_balance_anchor_weights(labels),
            num_samples=len(triplets),
            replacement=True,
            generator=loader_generator,
        )
    loader = DataLoader(
        TripletDataset(),
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate,
        generator=loader_generator if sampler is None else None,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, len(loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        positive_similarity = 0.0
        negative_similarity = 0.0
        seen = 0
        for encoded in loader:
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            optimizer.zero_grad(set_to_none=True)
            hidden = model(**encoded).last_hidden_state
            vectors = normalized_cls(hidden, encoded["attention_mask"], torch)
            anchor = vectors[0::3]
            positive = vectors[1::3]
            negative = vectors[2::3]
            positive_score = (anchor * positive).sum(dim=1)
            negative_score = (anchor * negative).sum(dim=1)
            loss = torch.relu(config.margin - positive_score + negative_score).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            batch_size = int(anchor.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            positive_similarity += float(positive_score.detach().sum().cpu())
            negative_similarity += float(negative_score.detach().sum().cpu())
            seen += batch_size
        history.append(
            {
                "epoch": epoch,
                "loss": round(total_loss / max(1, seen), 6),
                "positive_similarity": round(positive_similarity / max(1, seen), 6),
                "negative_similarity": round(negative_similarity / max(1, seen), 6),
            }
        )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "training_seconds": round(time.perf_counter() - started, 3),
        "trainable_fraction": round(trainable_parameters / max(1, total_parameters), 6),
        "peak_vram_mb": round(
            torch.cuda.max_memory_allocated(resolved_device) / 1024**2
            if resolved_device.type == "cuda"
            else 0.0,
            2,
        ),
    }


def encode_articles(
    model,
    tokenizer,
    articles: list[ArticleRecord],
    *,
    max_length: int,
    max_content_chars: int,
    batch_size: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    import torch

    resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = [
        build_production_text(article, max_content_chars=max_content_chars)
        for article in articles
    ]
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            vectors = normalized_cls(hidden, encoded["attention_mask"], torch)
            chunks.append(vectors.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)
