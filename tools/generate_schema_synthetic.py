from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from eventlens.config import load_settings
from eventlens.env import env_float, env_int, env_str, load_env_file
from eventlens.event_retrieval import EventSchemaIndex
from eventlens.io import read_competition_labeled_excel
from eventlens.llm_agent import OpenAICompatibleChatClient, _parse_action
from eventlens.preprocess import clean_text


def _label_definitions(schema: EventSchemaIndex, *, scope: str) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for row in schema.definitions:
        if row.scope != scope:
            continue
        bucket = output.setdefault(row.event_name, [])
        if row.description and row.description not in bucket:
            bucket.append(row.description)
    return output


def _valid_sample(sample: dict, *, label: str, forbidden_labels: list[str]) -> tuple[str, str] | None:
    title = clean_text(sample.get("title", ""))
    content = clean_text(sample.get("content", ""))
    merged = f"{title} {content}"
    if len(content) < 60 or not title:
        return None
    if label in merged:
        return None
    if any(name in merged for name in forbidden_labels if name != label):
        return None
    if any(token in merged for token in ("事件类型", "分类标签", "该标签", "本文属于")):
        return None
    return title[:120], content[:1200]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    parser.add_argument("--samples-per-label", type=int, default=8)
    parser.add_argument("--max-train-count", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    load_env_file()
    settings = load_settings()
    train = read_competition_labeled_excel(settings.paths.tagged_train)[f"{args.scope}_event"]
    counts = Counter(str(row.event_label) for row in train)
    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    definitions = _label_definitions(schema, scope=args.scope)
    target_labels = sorted(
        label for label, count in counts.items() if count <= args.max_train_count and label in definitions
    )
    all_labels = sorted(definitions)
    client = OpenAICompatibleChatClient(
        base_url=env_str("EVENTLENS_LLM_BASE_URL", settings.agent_expert.base_url),
        model=env_str("EVENTLENS_LLM_MODEL", settings.agent_expert.model),
        api_key_env=env_str("EVENTLENS_LLM_API_KEY_ENV", settings.agent_expert.api_key_env),
        temperature=0.0,
        max_tokens=max(1800, env_int("EVENTLENS_LLM_MAX_TOKENS", 2048)),
        timeout_seconds=env_float("EVENTLENS_LLM_TIMEOUT_SECONDS", 180.0),
        thinking="disabled",
        reasoning_effort="high",
        json_output=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.add(str(row.get("event_label", "")))

    with output_path.open("a", encoding="utf-8") as file:
        failed_labels: list[str] = []
        for label in target_labels:
            if label in existing:
                continue
            definition_text = "\n".join(definitions[label][:5])
            prompt = {
                "task": "generate_low_resource_financial_news_training_samples",
                "target_event_definition": definition_text,
                "sample_count": args.samples_per_label,
                "constraints": [
                    "生成中文金融新闻风格的标题和正文，事实要明确支撑目标事件定义",
                    "不要出现事件类型名称、分类标签、元数据解释或答案提示",
                    "不要使用真实上市公司名称、证券代码或真实人物姓名，可用某公司/某企业等自然表达",
                    "每条正文约120到350个中文字符，标题和叙述角度尽量多样",
                    "不要混入相邻但不同的事件，只描述一个主要事件",
                    "不得虚构为真实发生的具体公司新闻，保持合成训练样本属性",
                ],
                "forbidden_event_names": all_labels,
                "output_schema": {
                    "samples": [{"title": "string", "content": "string"}]
                },
            }
            messages = [
                {
                    "role": "system",
                    "content": "你是金融NLP数据增强器。只输出合法JSON，不输出解释或思维过程。",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ]
            payload = None
            for attempt in range(2):
                raw = client.complete(messages)
                try:
                    payload = _parse_action(raw)
                    break
                except ValueError:
                    messages.append(
                        {
                            "role": "user",
                            "content": "上一条不是合法JSON。严格按 output_schema 只返回一个JSON object。",
                        }
                    )
            if payload is None:
                failed_labels.append(label)
                continue
            samples = payload.get("samples") or []
            seen = set()
            accepted = []
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                valid = _valid_sample(sample, label=label, forbidden_labels=all_labels)
                if valid is None:
                    continue
                title, content = valid
                signature = clean_text(f"{title}{content}")
                if signature in seen:
                    continue
                seen.add(signature)
                accepted.append((title, content))
            schema_hash = hashlib.sha256(definition_text.encode("utf-8")).hexdigest()
            for rank, (title, content) in enumerate(accepted, start=1):
                row = {
                    "event_label": label,
                    "title": title,
                    "content": content,
                    "source": "synthetic_schema_deepseek",
                    "generation_rank": rank,
                    "generator_model": client.model,
                    "schema_hash": schema_hash,
                    "train_count": counts[label],
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()

    summary = {
        "scope": args.scope,
        "target_label_count": len(target_labels),
        "target_labels": target_labels,
        "samples_per_label_requested": args.samples_per_label,
        "failed_labels": failed_labels,
        "api_usage": client.usage,
        "output": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
