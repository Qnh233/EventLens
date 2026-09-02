from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol

from pydantic import BaseModel, Field

from eventlens.event_retrieval import RoutedArticleRecallResult
from eventlens.schema import ArticleRecord, EventPrediction
from eventlens.subject_routing import SubjectRouteResult


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatibleChatClient:
    """最小 OpenAI-compatible chat client，不引入额外 SDK。"""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_seconds: float = 60.0,
        thinking: str = "disabled",
        reasoning_effort: str = "high",
        json_output: bool = True,
    ):
        if not base_url:
            raise ValueError("外部 LLM provider 需要 base_url")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未配置")
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.json_output = json_output
        self.usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.json_output:
            body["response_format"] = {"type": "json_object"}
        if self.thinking == "enabled":
            body["thinking"] = {"type": "enabled"}
            body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = self.temperature
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM API 未返回 choices")
        usage = body.get("usage") or {}
        self.usage["requests"] += 1
        for key in (
            "prompt_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
            "completion_tokens",
            "total_tokens",
        ):
            self.usage[key] += int(usage.get(key) or 0)
        details = usage.get("completion_tokens_details") or {}
        self.usage["reasoning_tokens"] += int(details.get("reasoning_tokens") or 0)
        return str(choices[0].get("message", {}).get("content", ""))


class TransformersChatClient:
    """4090D POC provider；生产可无缝替换为 OpenAI-compatible API。"""

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        max_tokens: int = 256,
        cache_folder: str | None = None,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=cache_folder,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            cache_dir=cache_folder,
            local_files_only=local_files_only,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class AgentToolCall(BaseModel):
    step: int
    action: str
    result: str


class AgentDecision(BaseModel):
    article_id: str
    baseline_event: str
    final_event: str | None = None
    decision: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    valid: bool = True
    requested_collection: bool = False
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    steps: int = 0
    latency_ms: int = 0
    error: str = ""
    verifier_applied: bool = False
    verifier_accepted: bool | None = None
    verifier_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    verifier_reason: str = ""
    verifier_latency_ms: int = 0


class EventChangeVerifier:
    """第二角色：只审查 Expert 的改判，默认保守拒绝不充分的变更。"""

    def __init__(self, client: ChatClient, *, max_content_chars: int = 4000):
        self.client = client
        self.max_content_chars = max_content_chars

    def verify(
        self,
        article: ArticleRecord,
        prediction: EventPrediction,
        recall: RoutedArticleRecallResult,
        decision: AgentDecision,
    ) -> AgentDecision:
        if (
            not decision.valid
            or decision.final_event is None
            or decision.final_event == prediction.event_type
        ):
            return decision
        started = time.perf_counter()
        candidate = next(
            (row for row in recall.candidates if row.event_name == decision.final_event),
            None,
        )
        messages = [
            {"role": "system", "content": _VERIFIER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "baseline_event": prediction.event_type,
                        "proposed_event": decision.final_event,
                        "expert_reason": decision.reason,
                        "proposed_schema": candidate.description if candidate else "",
                        "title": article.title,
                        "source": article.source,
                        "content": (article.content or "")[: self.max_content_chars],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            payload = _parse_action(self.client.complete(messages))
            accepted = bool(payload.get("accept", False))
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0.0))))
            reason = str(payload.get("reason") or "")[:300]
        except (ValueError, TypeError):
            accepted = False
            confidence = 0.0
            reason = "Verifier 输出无效，按安全策略拒绝改判"
        latency_ms = round((time.perf_counter() - started) * 1000)
        if accepted:
            return decision.model_copy(
                update={
                    "verifier_applied": True,
                    "verifier_accepted": True,
                    "verifier_confidence": confidence,
                    "verifier_reason": reason,
                    "verifier_latency_ms": latency_ms,
                }
            )
        return decision.model_copy(
            update={
                "final_event": prediction.event_type,
                "decision": "verifier_reject_keep_baseline",
                "reason": f"Expert 提议被 Verifier 拒绝：{reason}",
                "verifier_applied": True,
                "verifier_accepted": False,
                "verifier_confidence": confidence,
                "verifier_reason": reason,
                "verifier_latency_ms": latency_ms,
            }
        )


class EventExpertAgent:
    """受约束事件专家：必须先查工具，最终只能选 baseline/BGE 候选或 abstain。"""

    def __init__(self, client: ChatClient, *, max_steps: int = 4, max_content_chars: int = 4000):
        self.client = client
        self.max_steps = max(2, max_steps)
        self.max_content_chars = max_content_chars

    def run(
        self,
        article: ArticleRecord,
        prediction: EventPrediction,
        route: SubjectRouteResult,
        recall: RoutedArticleRecallResult,
    ) -> AgentDecision:
        started = time.perf_counter()
        allowed_events = list(dict.fromkeys(
            [prediction.event_type, *[row.event_name for row in recall.candidates]]
        ))
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "review_financial_event",
                        "article_id": article.article_id,
                        "baseline_event": prediction.event_type,
                        "baseline_confidence": prediction.classifier_confidence,
                        "subject_method": route.method,
                        "subject_top1_margin": route.top1_margin,
                        "allowed_events": allowed_events,
                        "event_candidates": [
                            {"event_name": row.event_name, "score": row.score}
                            for row in recall.candidates
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        traces: list[AgentToolCall] = []
        inspected_schema = False
        inspected_evidence = False
        requested_collection = False
        last_error = ""

        for step in range(1, self.max_steps + 1):
            raw = self.client.complete(messages)
            try:
                action = _parse_action(raw)
            except ValueError as exc:
                last_error = str(exc)
                messages.append({
                    "role": "user",
                    "content": "上一条不是合法 JSON。只返回约定 JSON，不要 Markdown。",
                })
                continue

            name = str(action.get("action", ""))
            if name == "inspect_schema":
                inspected_schema = True
                result = json.dumps(
                    [
                        {
                            "event_name": row.event_name,
                            "description": row.description,
                            "subject": row.subject_name,
                            "score": row.score,
                        }
                        for row in recall.candidates
                    ],
                    ensure_ascii=False,
                )
            elif name == "inspect_evidence":
                inspected_evidence = True
                result = json.dumps(
                    {
                        "title": article.title,
                        "source": article.source,
                        "publish_time": article.publish_time.isoformat() if article.publish_time else None,
                        "content": (article.content or "")[: self.max_content_chars],
                        "baseline_evidence": prediction.evidence_sentence,
                    },
                    ensure_ascii=False,
                )
            elif name == "request_collection":
                requested_collection = True
                result = json.dumps(
                    {
                        "accepted": True,
                        "preferred_sources": ["官方", "公司公告", "主流财经媒体"],
                        "note": "当前 POC 只生成补采请求，不伪造外部证据",
                    },
                    ensure_ascii=False,
                )
            elif name == "final":
                decision_name = str(action.get("decision", "keep_baseline"))
                required_observation_ready = (
                    inspected_evidence
                    and (decision_name != "select_candidate" or inspected_schema)
                )
                if not required_observation_ready:
                    result = "final 前必须至少调用 inspect_schema 或 inspect_evidence"
                    traces.append(AgentToolCall(step=step, action="invalid_final", result=result))
                    messages.append({"role": "user", "content": f"TOOL_RESULT: {result}"})
                    continue
                decision = _validate_final(
                    article.article_id,
                    prediction.event_type,
                    allowed_events,
                    action,
                    requested_collection,
                    traces,
                    step,
                    started,
                )
                return decision
            else:
                result = "未知 action；允许 inspect_schema/inspect_evidence/request_collection/final"

            traces.append(AgentToolCall(step=step, action=name or "invalid", result=result[:1200]))
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"TOOL_RESULT: {result}"})

        return AgentDecision(
            article_id=article.article_id,
            baseline_event=prediction.event_type,
            final_event=prediction.event_type,
            decision="fallback_baseline",
            confidence=prediction.classifier_confidence,
            reason="Agent 未在最大步数内形成合法 final，安全回退 baseline",
            valid=False,
            requested_collection=requested_collection,
            tool_calls=traces,
            steps=self.max_steps,
            latency_ms=round((time.perf_counter() - started) * 1000),
            error=last_error or "max_steps_exceeded",
        )


def _parse_action(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("未找到 JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("JSON 解析失败") from exc
    if not isinstance(payload, dict):
        raise ValueError("Agent 输出必须是 JSON object")
    return payload


def _validate_final(
    article_id: str,
    baseline_event: str,
    allowed_events: list[str],
    action: dict,
    requested_collection: bool,
    traces: list[AgentToolCall],
    step: int,
    started: float,
) -> AgentDecision:
    decision = str(action.get("decision", "keep_baseline"))
    event_name = str(action.get("event_name") or "")
    reason = str(action.get("reason") or "")[:300]
    try:
        confidence = min(1.0, max(0.0, float(action.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    if decision == "abstain":
        final_event = None
    elif decision == "keep_baseline":
        final_event = baseline_event
    elif decision == "select_candidate" and event_name in allowed_events:
        final_event = event_name
    else:
        return AgentDecision(
            article_id=article_id,
            baseline_event=baseline_event,
            final_event=baseline_event,
            decision="fallback_baseline",
            confidence=0.0,
            reason="Agent final 超出允许事件集合，安全回退 baseline",
            valid=False,
            requested_collection=requested_collection,
            tool_calls=traces,
            steps=step,
            latency_ms=round((time.perf_counter() - started) * 1000),
            error="invalid_final_event",
        )
    return AgentDecision(
        article_id=article_id,
        baseline_event=baseline_event,
        final_event=final_event,
        decision=decision,
        confidence=confidence,
        reason=reason,
        valid=True,
        requested_collection=requested_collection,
        tool_calls=traces,
        steps=step,
        latency_ms=round((time.perf_counter() - started) * 1000),
    )


_SYSTEM_PROMPT = """你是 EventLens 的受约束金融事件 Expert Agent。
你不是自由聊天模型。你必须通过工具观察后再做最终判断，且不得输出候选集合外的新事件类型。
每次只能返回一个 JSON object，不要 Markdown，不要输出思维过程，只给简短决策理由。

允许 action：
1. {"action":"inspect_schema"}
2. {"action":"inspect_evidence"}
3. {"action":"request_collection","reason":"为什么证据不足"}
4. {"action":"final","decision":"keep_baseline|select_candidate|abstain","event_name":"候选事件或空","confidence":0到1,"reason":"简短可审计理由"}

规则：
- keep_baseline/abstain 前至少 inspect_evidence。
- select_candidate 改判前必须同时 inspect_evidence 和 inspect_schema。
- 当前 Event Expert 只接收已经 hard-route 的主体；不要重新猜主体。
- BGE 候选只是召回，不代表正确。只有正文直接支持候选定义时才改判；否则保持 baseline 或 abstain。
- 证据不足时优先 request_collection 或 abstain；不要猜测不存在的事实。"""


_VERIFIER_PROMPT = """你是 EventLens 的 Critic/Verifier Agent，只负责审查另一个 Expert Agent 的事件改判。
你的目标是降低误改判，而不是追求更多修改。
只有当原文直接支持 proposed_event 的 schema 定义，且 proposed_event 明显比 baseline_event 更贴合主要事件事实时才 accept=true。
如果只是主题相关、行业背景、推测、间接影响，或 baseline_event 仍然合理，必须 accept=false。
只返回 JSON：{"accept":true或false,"confidence":0到1,"reason":"不超过一句的可审计理由"}。不要输出思维过程，不要 Markdown。"""
