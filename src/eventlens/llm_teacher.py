from __future__ import annotations

import json
import time
from typing import Protocol

from pydantic import BaseModel, Field

from eventlens.schema import ArticleRecord


class TeacherChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class TeacherDecision(BaseModel):
    article_id: str
    event_label: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    abstain: bool = False
    reason: str = ""
    valid: bool = True
    latency_ms: int = 0
    error: str = ""


class TeacherVerificationDecision(BaseModel):
    accept: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    valid: bool = True
    latency_ms: int = 0
    error: str = ""


class CandidateEventTeacher:
    """只允许从候选事件中选择；用于受治理伪标签，不拥有生产最终决策权。"""

    def __init__(self, client: TeacherChatClient, *, max_content_chars: int = 4000):
        self.client = client
        self.max_content_chars = max_content_chars

    def label(
        self,
        article: ArticleRecord,
        candidates: list[dict[str, str]],
        *,
        exemplars: dict[str, list[dict[str, str]]] | None = None,
    ) -> TeacherDecision:
        started = time.perf_counter()
        allowed = [str(row["event_name"]) for row in candidates]
        prompt = _EXEMPLAR_TEACHER_PROMPT if exemplars else _TEACHER_PROMPT
        user_payload = {
            "title": article.title,
            "source": article.source,
            "content": (article.content or "")[: self.max_content_chars],
            "candidate_events": candidates,
        }
        if exemplars:
            user_payload["candidate_exemplars"] = exemplars
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]
        try:
            payload = json.loads(self.client.complete(messages))
            event = str(payload.get("event_label") or "").strip() or None
            abstain = bool(payload.get("abstain", False))
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
            reason = str(payload.get("reason") or "")[:300]
            if abstain:
                event = None
            elif event not in allowed:
                raise ValueError("LLM 返回了候选集合外事件")
            return TeacherDecision(
                article_id=article.article_id,
                event_label=event,
                confidence=confidence,
                abstain=abstain,
                reason=reason,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return TeacherDecision(
                article_id=article.article_id,
                valid=False,
                abstain=True,
                latency_ms=round((time.perf_counter() - started) * 1000),
                error=str(exc)[:300],
            )


class CandidateChangeVerifier:
    """只审查 Teacher 的改判；证据不足时默认保留基线。"""

    def __init__(self, client: TeacherChatClient, *, max_content_chars: int = 4000):
        self.client = client
        self.max_content_chars = max_content_chars

    def verify(
        self,
        article: ArticleRecord,
        *,
        baseline_event: str,
        proposed_event: str,
        teacher_reason: str,
        candidate_definitions: dict[str, str],
        exemplars: dict[str, list[dict[str, str]]] | None = None,
    ) -> TeacherVerificationDecision:
        started = time.perf_counter()
        messages = [
            {"role": "system", "content": _CHANGE_VERIFIER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "baseline_event": baseline_event,
                        "proposed_event": proposed_event,
                        "teacher_reason": teacher_reason,
                        "candidate_definitions": candidate_definitions,
                        "candidate_exemplars": exemplars or {},
                        "title": article.title,
                        "source": article.source,
                        "content": (article.content or "")[: self.max_content_chars],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            payload = json.loads(self.client.complete(messages))
            return TeacherVerificationDecision(
                accept=bool(payload.get("accept", False)),
                confidence=min(1.0, max(0.0, float(payload.get("confidence", 0.0)))),
                reason=str(payload.get("reason") or "")[:300],
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return TeacherVerificationDecision(
                accept=False,
                valid=False,
                latency_ms=round((time.perf_counter() - started) * 1000),
                error=str(exc)[:300],
            )


_TEACHER_PROMPT = """你是金融事件标注教师。任务是依据新闻主要事实，从给定候选事件中选择最准确的一个标签。
规则：
1. 只判断新闻主要事实，不因为公司名称、行业背景或次要提及猜标签。
2. 必须严格理解每个候选事件的定义；不能输出候选集合外标签。
3. 如果新闻证据不足以在候选之间可靠区分，必须 abstain=true，不能强猜。
4. confidence 表示你对标签正确性的概率判断，0~1；只有直接、明确的事实支持才给高置信。
5. 只返回 JSON，不要 Markdown：
{"event_label":"候选事件名或空字符串","confidence":0.0,"abstain":false,"reason":"简短依据"}
"""


_EXEMPLAR_TEACHER_PROMPT = """你是金融事件标注教师。任务是依据新闻主要事实，从给定候选事件中选择最准确的一个标签。
candidate_exemplars 是从训练 Gold 中检索出的同类真实历史新闻，只用于帮助理解候选标签边界，不代表当前新闻一定属于该类。
规则：
1. 先比较当前新闻与各候选定义及 exemplar 的关键事实差异，尤其区分语义相近候选；不能仅因词面相似复制 exemplar 标签。
2. 只判断当前新闻的主要事实，不因为公司名称、行业背景或次要提及猜标签。
3. 只能输出 candidate_events 中的事件；如果证据不足以可靠区分，必须 abstain=true。
4. reason 必须指出支持所选标签的当前新闻事实，并简述它与最接近竞争标签的关键区别。
5. confidence 表示标签正确概率，0~1；只有当前新闻直接事实和候选边界都明确时才给高置信。
6. 只返回 JSON，不要 Markdown：
{"event_label":"候选事件名或空字符串","confidence":0.0,"abstain":false,"reason":"当前事实 + 与竞争标签的区别"}
"""


_CHANGE_VERIFIER_PROMPT = """你是金融事件改判审核员。你不重新做全量分类，只判断 proposed_event 是否有足够直接证据推翻 baseline_event。
规则：
1. 默认拒绝改判。只有当前新闻主要事实明显符合 proposed_event，且与 baseline_event 的核心定义存在清晰可解释差异时才能 accept=true。
2. candidate_exemplars 只帮助理解标签边界，不能替代当前新闻证据；不能因为词面相似就接受改判。
3. 若 baseline_event 与 proposed_event 都合理、事实不足、主要事实不清晰或需要背景推断，必须 accept=false。
4. reason 必须说明当前新闻中支持或不足以支持改判的具体事实。
5. 只返回 JSON，不要 Markdown：
{"accept":false,"confidence":0.0,"reason":"简短审核依据"}
"""
