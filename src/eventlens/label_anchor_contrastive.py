from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eventlens.preprocess import clean_text
from eventlens.schema import ArticleRecord


@dataclass(frozen=True)
class LabelAnchorTrainingConfig:
    model: str
    max_length: int = 512
    max_content_chars: int = 2400
    batch_size: int = 8
    epochs: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    temperature: float = 0.07
    class_weight_power: float = 0.5
    trainable_last_layers: int = 2
    hard_negative_weight: float = 0.0
    hard_negative_margin: float = 0.1
    random_state: int = 42


def build_production_text(
    article: ArticleRecord,
    *,
    max_content_chars: int = 2400,
) -> str:
    """Production-like text: deliberately excludes labeled-only subject fields."""

    parts = [
        f"标题：{clean_text(article.title)}",
        f"来源：{clean_text(article.source)}",
        f"正文：{clean_text(article.content)[:max_content_chars]}",
    ]
    return " ".join(part for part in parts if part.split("：", 1)[-1])


def class_weights(
    labels: list[str],
    classes: list[str],
    *,
    power: float,
) -> np.ndarray:
    if power < 0:
        raise ValueError("class_weight_power must be >= 0")
    counts = Counter(labels)
    if any(counts[label] <= 0 for label in classes):
        raise ValueError("every class must occur in training labels")
    raw = np.asarray(
        [(len(labels) / (len(classes) * counts[label])) ** power for label in classes],
        dtype=np.float32,
    )
    return raw / max(float(raw.mean()), 1e-12)


def _encoder_layers(model) -> list:
    candidates = [
        getattr(getattr(model, "encoder", None), "layer", None),
        getattr(getattr(getattr(model, "roberta", None), "encoder", None), "layer", None),
        getattr(getattr(getattr(model, "bert", None), "encoder", None), "layer", None),
        getattr(getattr(getattr(model, "xlm_roberta", None), "encoder", None), "layer", None),
    ]
    for layers in candidates:
        if layers is not None:
            return list(layers)
    raise ValueError("unsupported encoder architecture: cannot find encoder.layer")


def freeze_except_last_layers(model, *, last_n: int) -> tuple[int, int]:
    if last_n <= 0:
        raise ValueError("last_n must be >= 1")
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = _encoder_layers(model)
    for layer in layers[-min(last_n, len(layers)) :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    # Keep the final layer norm trainable when present. This is a tiny, stable
    # adaptation surface and avoids changing embeddings through a new head.
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if "layernorm" in lowered or "layer_norm" in lowered:
            if any(f"layer.{index}." in name for index in range(max(0, len(layers) - last_n), len(layers))):
                parameter.requires_grad = True
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def normalized_cls(last_hidden_state, attention_mask, torch_module):
    # BGE-M3's sentence-transformers config uses CLS pooling followed by L2 norm.
    embedding = last_hidden_state[:, 0]
    return torch_module.nn.functional.normalize(embedding, p=2, dim=-1)


def _encode_texts(
    model,
    tokenizer,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int,
    device,
    torch_module,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch_module.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            embeddings = normalized_cls(hidden, encoded["attention_mask"], torch_module)
            output.append(embeddings.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0) if output else np.empty((0, 0), dtype=np.float32)


def train_label_anchor_encoder(
    train_articles: list[ArticleRecord],
    train_labels: list[str],
    *,
    label_texts: list[str],
    label_names: list[str],
    config: LabelAnchorTrainingConfig,
    hard_negative_ids_by_class: dict[int, list[int]] | None = None,
    output_dir: str | Path | None = None,
    local_files_only: bool = True,
    device: str = "cuda",
) -> dict:
    if len(train_articles) != len(train_labels):
        raise ValueError("train article/label count mismatch")
    if len(label_texts) != len(label_names) or not label_names:
        raise ValueError("label anchor text/name mismatch")
    if set(train_labels) - set(label_names):
        raise ValueError("training labels missing from label anchors")

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("label-anchor training requires torch + transformers") from exc

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(config.model, local_files_only=local_files_only)
    model.to(resolved_device)

    # Anchor vectors are fixed from the original BGE representation. The task
    # encoder learns to move articles toward semantic label anchors without
    # allowing scarce Gold data to distort label meanings themselves.
    anchor_vectors = _encode_texts(
        model,
        tokenizer,
        label_texts,
        max_length=config.max_length,
        batch_size=max(1, min(config.batch_size, len(label_texts))),
        device=resolved_device,
        torch_module=torch,
    )
    anchor_tensor = torch.tensor(anchor_vectors, dtype=torch.float32, device=resolved_device)
    trainable_parameters, total_parameters = freeze_except_last_layers(
        model, last_n=config.trainable_last_layers
    )

    class_to_id = {label: index for index, label in enumerate(label_names)}
    weights = class_weights(train_labels, label_names, power=config.class_weight_power)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=resolved_device)
    texts = [
        build_production_text(article, max_content_chars=config.max_content_chars)
        for article in train_articles
    ]

    class TextDataset(Dataset):
        def __len__(self) -> int:
            return len(texts)

        def __getitem__(self, index: int):
            return texts[index], class_to_id[train_labels[index]]

    def collate(rows):
        batch_texts = [row[0] for row in rows]
        batch_labels = torch.tensor([row[1] for row in rows], dtype=torch.long)
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )
        return encoded, batch_labels

    loader = DataLoader(
        TextDataset(),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(config.random_state),
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
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    hard_negative_tensor = None
    if config.hard_negative_weight > 0:
        if not hard_negative_ids_by_class:
            raise ValueError("hard-negative training requires hard_negative_ids_by_class")
        rows: list[list[int]] = []
        expected_width: int | None = None
        for class_id in range(len(label_names)):
            negatives = [int(value) for value in hard_negative_ids_by_class.get(class_id, [])]
            if not negatives:
                raise ValueError(f"missing hard negatives for class {class_id}")
            if class_id in negatives:
                raise ValueError("hard negatives must not contain the positive class")
            expected_width = len(negatives) if expected_width is None else expected_width
            if len(negatives) != expected_width:
                raise ValueError("hard-negative lists must have equal width")
            rows.append(negatives)
        hard_negative_tensor = torch.tensor(rows, dtype=torch.long, device=resolved_device)
    history: list[dict] = []
    started = time.perf_counter()
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for encoded, target in loader:
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            target = target.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            hidden = model(**encoded).last_hidden_state
            article_vectors = normalized_cls(hidden, encoded["attention_mask"], torch)
            logits = article_vectors @ anchor_tensor.T / config.temperature
            loss = loss_fn(logits, target)
            if hard_negative_tensor is not None:
                negative_ids = hard_negative_tensor[target]
                negative_logits = torch.gather(logits, 1, negative_ids)
                positive_logits = torch.gather(logits, 1, target[:, None])
                margin_loss = torch.relu(
                    config.hard_negative_margin - positive_logits + negative_logits
                ).mean()
                loss = loss + config.hard_negative_weight * margin_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            correct += int((logits.argmax(dim=-1) == target).sum().item())
            seen += int(target.numel())
        history.append(
            {
                "epoch": epoch,
                "loss": round(total_loss / max(1, len(loader)), 6),
                "train_accuracy": round(correct / max(1, seen), 6),
            }
        )
    training_seconds = time.perf_counter() - started
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(resolved_device) / 1024**2
        if resolved_device.type == "cuda"
        else 0.0
    )
    if output_dir:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination)
        tokenizer.save_pretrained(destination)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "anchor_vectors": anchor_vectors,
        "training_seconds": round(training_seconds, 3),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "trainable_fraction": round(trainable_parameters / max(1, total_parameters), 6),
        "history": history,
    }


def predict_label_anchor_scores(
    model,
    tokenizer,
    articles: list[ArticleRecord],
    *,
    anchor_vectors: np.ndarray,
    config: LabelAnchorTrainingConfig,
    device: str = "cuda",
) -> np.ndarray:
    import torch

    resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    texts = [
        build_production_text(article, max_content_chars=config.max_content_chars)
        for article in articles
    ]
    vectors = _encode_texts(
        model,
        tokenizer,
        texts,
        max_length=config.max_length,
        batch_size=config.batch_size,
        device=resolved_device,
        torch_module=torch,
    )
    anchors = np.asarray(anchor_vectors, dtype=np.float32)
    return vectors @ anchors.T / config.temperature

