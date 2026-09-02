from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from eventlens.preprocess import clean_text
from eventlens.schema import ArticleRecord


class TransformerClassificationMetrics(BaseModel):
    sample_count: int
    accuracy: float
    macro_f1: float


class TransformerEpochMetrics(BaseModel):
    epoch: int
    train_loss: float
    validation_accuracy: float
    validation_macro_f1: float


class TransformerEventExperimentReport(BaseModel):
    scope: str
    model: str
    train_count: int
    validation_count: int
    external_count: int
    label_count: int
    best_epoch: int
    best_validation: TransformerClassificationMetrics
    external: TransformerClassificationMetrics
    baseline_external_macro_f1: float
    macro_f1_gain: float
    gate_macro_f1: float
    gate_passed: bool
    training_seconds: float
    external_prediction_seconds: float
    peak_vram_mb: float
    history: list[TransformerEpochMetrics] = Field(default_factory=list)


@dataclass(frozen=True)
class TransformerTrainingConfig:
    model: str
    max_length: int = 384
    max_content_chars: int = 1800
    batch_size: int = 16
    epochs: int = 6
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    validation_ratio: float = 0.2
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.05
    class_weight_power: float = 0.5
    early_stopping_patience: int = 2
    include_subject_fields: bool = False
    gate_macro_f1: float = 0.78
    random_state: int = 42


def build_transformer_text(
    article: ArticleRecord,
    *,
    max_content_chars: int,
    include_subject_fields: bool,
) -> str:
    parts = [
        f"标题：{clean_text(article.title)}",
        f"来源：{clean_text(article.source)}",
    ]
    if include_subject_fields:
        if article.entity:
            parts.append(f"公司：{clean_text(article.entity)}")
        if article.industry:
            parts.append(f"行业：{clean_text(article.industry)}")
    parts.append(f"正文：{clean_text(article.content)[:max_content_chars]}")
    return " ".join(part for part in parts if part.split("：", 1)[-1])


def stratified_train_validation_indices(
    labels: list[str],
    *,
    validation_ratio: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    if not labels:
        raise ValueError("Transformer 训练至少需要 1 条标签样本")
    counts = Counter(labels)
    if min(counts.values()) < 2:
        raise ValueError("分层验证要求每个事件标签至少有 2 条训练样本")
    indices = list(range(len(labels)))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_ratio,
        random_state=random_state,
        stratify=labels,
    )
    return sorted(train_indices), sorted(validation_indices)


def classification_metrics(
    truth: list[str], predictions: list[str]
) -> TransformerClassificationMetrics:
    if len(truth) != len(predictions):
        raise ValueError("分类真值与预测数量不一致")
    return TransformerClassificationMetrics(
        sample_count=len(truth),
        accuracy=round(accuracy_score(truth, predictions), 6),
        macro_f1=round(
            f1_score(truth, predictions, average="macro", zero_division=0), 6
        ),
    )


def run_transformer_event_experiment(
    train_articles: list[ArticleRecord],
    external_articles: list[ArticleRecord],
    *,
    scope: str,
    config: TransformerTrainingConfig,
    baseline_external_macro_f1: float,
    output_model_dir: str | Path | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    device: str = "cuda",
) -> TransformerEventExperimentReport:
    """监督微调一个 encoder，内部验证选 epoch，外部集只做最终一次评测。"""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:  # pragma: no cover - GPU 环境能力检查
        raise RuntimeError("Transformer 实验需要 torch + transformers") from exc

    train_samples = [article for article in train_articles if article.event_label]
    external_samples = [article for article in external_articles if article.event_label]
    if len(train_samples) != len(train_articles):
        raise ValueError("Transformer 训练集存在缺失 event_label")
    if len(external_samples) != len(external_articles):
        raise ValueError("Transformer 外部评测集存在缺失 event_label")

    labels = sorted({str(article.event_label) for article in train_samples})
    unknown_external = sorted(
        {str(article.event_label) for article in external_samples} - set(labels)
    )
    if unknown_external:
        raise ValueError(f"外部评测包含训练未见事件标签: {unknown_external}")
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    train_labels = [str(article.event_label) for article in train_samples]
    train_indices, validation_indices = stratified_train_validation_indices(
        train_labels,
        validation_ratio=config.validation_ratio,
        random_state=config.random_state,
    )
    texts = [
        build_transformer_text(
            article,
            max_content_chars=config.max_content_chars,
            include_subject_fields=config.include_subject_fields,
        )
        for article in train_samples
    ]
    external_texts = [
        build_transformer_text(
            article,
            max_content_chars=config.max_content_chars,
            include_subject_fields=config.include_subject_fields,
        )
        for article in external_samples
    ]

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    resolved_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    model.to(resolved_device)

    class EncodedDataset(Dataset):
        def __init__(self, rows: list[int], all_texts: list[str], all_labels: list[str]):
            self.rows = rows
            self.all_texts = all_texts
            self.all_labels = all_labels

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            row = self.rows[index]
            encoded = tokenizer(
                self.all_texts[row],
                truncation=True,
                max_length=config.max_length,
            )
            encoded["labels"] = label_to_id[self.all_labels[row]]
            return encoded

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    generator = torch.Generator().manual_seed(config.random_state)
    train_loader = DataLoader(
        EncodedDataset(train_indices, texts, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        EncodedDataset(validation_indices, texts, train_labels),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    counts = Counter(train_labels[index] for index in train_indices)
    class_weights = []
    for label in labels:
        count = max(1, counts[label])
        weight = (len(train_indices) / (len(labels) * count)) ** config.class_weight_power
        class_weights.append(weight)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=resolved_device)
    weight_tensor = weight_tensor / weight_tensor.mean()
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    history: list[TransformerEpochMetrics] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: TransformerClassificationMetrics | None = None
    best_epoch = 0
    patience = 0
    training_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            labels_tensor = batch.pop("labels").to(resolved_device)
            inputs = {key: value.to(resolved_device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())

        validation_truth = [train_labels[index] for index in validation_indices]
        validation_predictions = _predict_labels(
            model,
            tokenizer,
            [texts[index] for index in validation_indices],
            id_to_label,
            max_length=config.max_length,
            batch_size=config.batch_size,
            device=resolved_device,
            collator=collator,
            torch_module=torch,
        )
        validation_metrics = classification_metrics(
            validation_truth, validation_predictions
        )
        history.append(
            TransformerEpochMetrics(
                epoch=epoch,
                train_loss=round(total_loss / max(1, len(train_loader)), 6),
                validation_accuracy=validation_metrics.accuracy,
                validation_macro_f1=validation_metrics.macro_f1,
            )
        )
        if best_validation is None or validation_metrics.macro_f1 > best_validation.macro_f1:
            best_validation = validation_metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    training_seconds = time.perf_counter() - training_started

    if best_state is None or best_validation is None:
        raise RuntimeError("Transformer 训练未产生有效 checkpoint")
    model.load_state_dict(best_state)
    model.to(resolved_device)

    prediction_started = time.perf_counter()
    external_predictions = _predict_labels(
        model,
        tokenizer,
        external_texts,
        id_to_label,
        max_length=config.max_length,
        batch_size=config.batch_size,
        device=resolved_device,
        collator=collator,
        torch_module=torch,
    )
    external_prediction_seconds = time.perf_counter() - prediction_started
    external_metrics = classification_metrics(
        [str(article.event_label) for article in external_samples],
        external_predictions,
    )
    peak_vram_mb = (
        torch.cuda.max_memory_allocated(resolved_device) / 1024**2
        if resolved_device.type == "cuda"
        else 0.0
    )

    if output_model_dir:
        model_dir = Path(output_model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    return TransformerEventExperimentReport(
        scope=scope,
        model=config.model,
        train_count=len(train_indices),
        validation_count=len(validation_indices),
        external_count=len(external_samples),
        label_count=len(labels),
        best_epoch=best_epoch,
        best_validation=best_validation,
        external=external_metrics,
        baseline_external_macro_f1=baseline_external_macro_f1,
        macro_f1_gain=round(
            external_metrics.macro_f1 - baseline_external_macro_f1, 6
        ),
        gate_macro_f1=config.gate_macro_f1,
        gate_passed=external_metrics.macro_f1 >= config.gate_macro_f1,
        training_seconds=round(training_seconds, 3),
        external_prediction_seconds=round(external_prediction_seconds, 3),
        peak_vram_mb=round(peak_vram_mb, 1),
        history=history,
    )


def _predict_labels(
    model,
    tokenizer,
    texts: list[str],
    id_to_label: dict[int, str],
    *,
    max_length: int,
    batch_size: int,
    device,
    collator,
    torch_module,
) -> list[str]:
    model.eval()
    predictions: list[str] = []
    with torch_module.no_grad():
        for start in range(0, len(texts), batch_size):
            rows = [
                tokenizer(text, truncation=True, max_length=max_length)
                for text in texts[start : start + batch_size]
            ]
            batch = collator(rows)
            inputs = {key: value.to(device) for key, value in batch.items()}
            predicted_ids = model(**inputs).logits.argmax(dim=-1).detach().cpu().tolist()
            predictions.extend(id_to_label[int(index)] for index in predicted_ids)
    return predictions
