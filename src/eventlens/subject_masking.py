from __future__ import annotations

import re

from eventlens.event_retrieval import EventSchemaIndex
from eventlens.preprocess import clean_text


SUBJECT_TOKEN = "主体占位"


def public_subject_aliases(schema: EventSchemaIndex, *, scope: str) -> list[str]:
    """只从公开事件 schema 生成推理阶段可获得的主体别名。"""

    aliases = {
        clean_text(definition.subject_name)
        for definition in schema.definitions
        if definition.scope == scope and len(clean_text(definition.subject_name)) >= 2
    }
    return sorted(aliases, key=lambda value: (-len(value), value))


def mask_public_subject_names(text: str, aliases: list[str]) -> str:
    masked = clean_text(text)
    for alias in aliases:
        masked = masked.replace(alias, SUBJECT_TOKEN)
    return re.sub(rf"(?:{SUBJECT_TOKEN}\s*)+", SUBJECT_TOKEN, masked).strip()
