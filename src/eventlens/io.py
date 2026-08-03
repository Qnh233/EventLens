from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from eventlens.preprocess import clean_text, normalize_entity, parse_datetime
from eventlens.schema import ArticleRecord

COLUMN_ALIASES: dict[str, set[str]] = {
    "article_id": {"article_id", "id", "文章id", "文章ID", "资讯id", "资讯ID"},
    "title": {"title", "文章标题", "标题"},
    "publish_time": {"publish_time", "publish_date", "date", "发布日期", "发布时间", "时间"},
    "source": {"source", "source_site", "来源网站", "来源", "媒体"},
    "content": {"content", "正文文本", "正文", "文章内容", "文本"},
    "entity": {"entity", "实体", "公司", "关联企业", "事件关联企业"},
    "industry": {"industry", "行业", "板块"},
    "event_label": {"event", "event_label", "事件", "事件类型", "事件类型名称"},
    "polarity_label": {"polarity", "event_polarity", "polarity_label", "事件情感正负面", "情感", "影响方向"},
    "impact_analysis": {"impact_analysis", "事件影响分析", "影响分析"},
    "duplicate_flag": {"duplicate_flag", "重复性标志", "是否重复", "重复标志"},
}


def _canonical_column(name: Any) -> str | None:
    raw = clean_text(name)
    lowered = raw.lower()
    for canonical, aliases in COLUMN_ALIASES.items():
        if raw in aliases or lowered in aliases:
            return canonical
    return None


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    used: set[str] = set()
    for column in df.columns:
        canonical = _canonical_column(column)
        if canonical and canonical not in used:
            rename_map[column] = canonical
            used.add(canonical)
    return df.rename(columns=rename_map)


def read_articles_excel(path: str | Path) -> list[ArticleRecord]:
    df = normalize_dataframe_columns(pd.read_excel(path))
    if "article_id" not in df.columns:
        df["article_id"] = [f"ROW-{idx + 1}" for idx in range(len(df))]

    articles: list[ArticleRecord] = []
    for idx, row in df.iterrows():
        extra = {str(k): row[k] for k in df.columns if k not in COLUMN_ALIASES}
        article = ArticleRecord(
            article_id=clean_text(row.get("article_id")) or f"ROW-{idx + 1}",
            title=clean_text(row.get("title")),
            publish_time=parse_datetime(row.get("publish_time")),
            source=clean_text(row.get("source")),
            content=clean_text(row.get("content")),
            entity=normalize_entity(row.get("entity")),
            industry=clean_text(row.get("industry")),
            event_label=clean_text(row.get("event_label")) or None,
            polarity_label=clean_text(row.get("polarity_label")) or None,
            impact_analysis=clean_text(row.get("impact_analysis")) or None,
            duplicate_flag=row.get("duplicate_flag"),
            source_row=int(idx) + 2,
            extra=extra,
        )
        articles.append(article)
    return articles


def profile_articles(articles: list[ArticleRecord]) -> dict[str, Any]:
    event_counter = Counter(a.event_label or "未标注" for a in articles)
    source_counter = Counter(a.source or "未知来源" for a in articles)
    missing_content = sum(1 for a in articles if not a.content)
    missing_entity = sum(1 for a in articles if not a.entity)
    return {
        "article_count": len(articles),
        "missing_content": missing_content,
        "missing_entity": missing_entity,
        "event_distribution": dict(event_counter.most_common()),
        "top_sources": dict(source_counter.most_common(20)),
    }


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            data = row.model_dump() if hasattr(row, "model_dump") else row
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def render_profile_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# 数据画像",
        "",
        "## 决策理由",
        "赛题评分依赖结构化字段，先确认字段缺失、标签分布和来源分布，避免盲目建模。",
        "",
        f"- 样本数：{profile['article_count']}",
        f"- 正文缺失：{profile['missing_content']}",
        f"- 实体缺失：{profile['missing_entity']}",
        "",
        "## 事件标签分布",
    ]
    lines.extend(f"- {k}: {v}" for k, v in profile["event_distribution"].items())
    lines.extend(["", "## 来源 Top20"])
    lines.extend(f"- {k}: {v}" for k, v in profile["top_sources"].items())
    return "\n".join(lines) + "\n"

