from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

import pandas as pd

from eventlens.schema import ArticleRecord

SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
COMPANY_SUFFIX_RE = re.compile(r"(股份有限公司|有限责任公司|有限公司|集团股份|集团|公司)$")

POSITIVE_WORDS = ("利好", "增长", "突破", "中标", "盈利", "增持", "通过", "获批")
NEGATIVE_WORDS = ("处罚", "违规", "亏损", "下滑", "诉讼", "问询", "立案", "退市", "减持", "风险")


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).replace("\u3000", " ")
    return SPACE_RE.sub(" ", text).strip()


def parse_datetime(value: Any) -> datetime | None:
    if is_missing(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def normalize_entity(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    first = re.split(r"[,，、;/；\s]+", text)[0]
    return COMPANY_SUFFIX_RE.sub("", first).strip()


def normalize_polarity(value: Any, fallback_text: str = "") -> str:
    text = clean_text(value)
    if text in {"正面", "负面", "中性"}:
        return text
    if text.lower() in {"positive", "pos", "利好"}:
        return "正面"
    if text.lower() in {"negative", "neg", "利空"}:
        return "负面"
    haystack = fallback_text or text
    if any(word in haystack for word in NEGATIVE_WORDS):
        return "负面"
    if any(word in haystack for word in POSITIVE_WORDS):
        return "正面"
    return "中性"


def build_model_text(article: ArticleRecord, max_content_chars: int = 1200) -> str:
    parts = [
        article.title,
        article.entity,
        article.industry,
        article.source,
        article.content[:max_content_chars],
    ]
    return " ".join(clean_text(part) for part in parts if clean_text(part))


def evidence_sentence(article: ArticleRecord, event_type: str = "") -> str:
    text = clean_text(article.content)
    if not text:
        return clean_text(article.title)[:80]
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return text[:80]
    keywords = [event_type, *NEGATIVE_WORDS, *POSITIVE_WORDS]
    for sentence in sentences:
        if any(word and word in sentence for word in keywords):
            return sentence[:80]
    return sentences[0][:80]


def text_tokens(text: str) -> set[str]:
    cleaned = clean_text(text)
    return {tok for tok in re.split(r"\W+", cleaned) if len(tok) >= 2}

