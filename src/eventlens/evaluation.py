from __future__ import annotations

import gc
import random
from collections import Counter
from pathlib import Path
from typing import Callable

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from eventlens.baseline import train_baseline
from eventlens.config import load_settings
from eventlens.schema import ArticleRecord


def _split_by_key(
    articles: list[ArticleRecord],
    key_fn: Callable[[ArticleRecord], str],
    holdout_ratio: float | None = None,
    seed: int | None = None,
) -> tuple[list[ArticleRecord], list[ArticleRecord]]:
    evaluation = load_settings().evaluation
    ratio = holdout_ratio if holdout_ratio is not None else evaluation.group_holdout_ratio
    random_seed = seed if seed is not None else evaluation.random_state
    groups = sorted({key_fn(article) for article in articles})
    rng = random.Random(random_seed)
    rng.shuffle(groups)
    holdout_size = max(1, min(len(groups) - 1, int(len(groups) * ratio))) if len(groups) > 1 else len(groups)
    holdout_groups = set(groups[:holdout_size])
    train = [article for article in articles if key_fn(article) not in holdout_groups]
    valid = [article for article in articles if key_fn(article) in holdout_groups]
    return train, valid


def time_split(
    articles: list[ArticleRecord],
    train_ratio: float | None = None,
) -> tuple[list[ArticleRecord], list[ArticleRecord]]:
    ratio = train_ratio if train_ratio is not None else load_settings().evaluation.time_train_ratio
    dated = sorted(articles, key=lambda article: (article.publish_time is None, article.publish_time))
    cut = max(1, min(len(dated) - 1, int(len(dated) * ratio))) if len(dated) > 1 else len(dated)
    return dated[:cut], dated[cut:]


def group_holdout_split(
    articles: list[ArticleRecord],
    group_field: str,
    holdout_ratio: float | None = None,
    seed: int | None = None,
) -> tuple[list[ArticleRecord], list[ArticleRecord]]:
    return _split_by_key(
        articles,
        lambda article: str(getattr(article, group_field) or "UNKNOWN"),
        holdout_ratio=holdout_ratio,
        seed=seed,
    )


def duplication_holdout_split(
    articles: list[ArticleRecord],
    holdout_ratio: float | None = None,
    seed: int | None = None,
) -> tuple[list[ArticleRecord], list[ArticleRecord]]:
    """相同 duplication_id 永不跨集合；未分组文章各自作为独立组。"""

    return _split_by_key(
        articles,
        lambda article: (
            f"dup:{article.duplication_id}"
            if article.duplication_id
            else f"article:{article.article_id}"
        ),
        holdout_ratio=holdout_ratio,
        seed=seed,
    )


def evaluate_split(
    train: list[ArticleRecord],
    valid: list[ArticleRecord],
) -> dict:
    train_labeled = [article for article in train if article.event_label]
    valid_labeled = [article for article in valid if article.event_label]
    if not train_labeled or not valid_labeled:
        return {"status": "skipped", "reason": "训练集或验证集缺少事件标签"}

    event_only_train = [
        article.model_copy(update={"polarity_label": None}) for article in train_labeled
    ]
    model = train_baseline(event_only_train)
    predictions = [model.predict_one(article) for article in valid_labeled]
    y_true = [str(article.event_label) for article in valid_labeled]
    y_pred = [prediction.event_type for prediction in predictions]
    labels = sorted(set(y_true) | set(y_pred))
    train_labels = {str(article.event_label) for article in train_labeled}
    unseen_valid_labels = sorted(set(y_true) - train_labels)
    closed_set_count = sum(label in train_labels for label in y_true)

    result = {
        "status": "ok",
        "train_size": len(train_labeled),
        "valid_size": len(valid_labeled),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "closed_set_coverage": closed_set_count / len(y_true),
        "unseen_valid_labels": unseen_valid_labels,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    del model, predictions
    gc.collect()
    return result


def render_generalization_report(articles: list[ArticleRecord]) -> str:
    labeled = [article for article in articles if article.event_label]
    event_counter = Counter(article.event_label or "未标注" for article in articles)
    is_industry_task = bool(labeled) and sum(
        article.task_scope.startswith("industry") or (not article.entity and bool(article.industry))
        for article in labeled
    ) >= len(labeled) / 2
    subject_field = "industry" if is_industry_task else "entity"
    subject_name = "行业" if is_industry_task else "公司"
    splitters: list[tuple[str, Callable[[list[ArticleRecord]], tuple[list[ArticleRecord], list[ArticleRecord]]]]] = [
        ("时间切分", time_split),
        (f"{subject_name}切分", lambda rows: group_holdout_split(rows, subject_field)),
        ("来源切分", lambda rows: group_holdout_split(rows, "source")),
        ("同源分组切分", duplication_holdout_split),
    ]

    lines = [
        "# 泛化评测报告",
        "",
        "## 决策理由",
        "真实 B 榜数据来源、公司和时间分布未知，随机切分容易高估效果，因此使用时间、公司、来源三类隔离切分并实际训练评测。",
        "",
        f"- 总样本数：{len(articles)}",
        f"- 有事件标签样本数：{len(labeled)}",
        "",
        "## 标签分布",
    ]
    lines.extend(f"- {label}: {count}" for label, count in event_counter.most_common())

    for name, splitter in splitters:
        train, valid = splitter(labeled)
        result = evaluate_split(train, valid)
        lines.extend(["", f"## {name}"])
        if result["status"] != "ok":
            lines.append(f"- 状态：跳过（{result['reason']}）")
            continue
        lines.extend(
            [
                f"- train={result['train_size']}, valid={result['valid_size']}",
                f"- Accuracy：{result['accuracy']:.4f}",
                f"- Macro-F1：{result['macro_f1']:.4f}",
                f"- 验证样本闭集覆盖率：{result['closed_set_coverage']:.2%}",
                f"- 训练中未出现的验证标签：{result['unseen_valid_labels'] or '无'}",
                "",
                "### 混淆矩阵",
                "",
                "| 实际\\预测 | " + " | ".join(result["labels"]) + " |",
                "|---|" + "---|" * len(result["labels"]),
            ]
        )
        for label, row in zip(result["labels"], result["confusion_matrix"]):
            lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")

        lines.extend(["", "### 分类指标", "", "| 类别 | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"])
        report = result["classification_report"]
        for label in result["labels"]:
            metrics = report[label]
            lines.append(
                f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1-score']:.4f} | {int(metrics['support'])} |"
            )
    return "\n".join(lines) + "\n"


def write_generalization_report(path: str | Path, articles: list[ArticleRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_generalization_report(articles), encoding="utf-8")
