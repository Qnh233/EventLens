from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventlens.config import load_settings
from eventlens.event_retrieval import EventSchemaIndex
from eventlens.learning import load_feedback_jsonl
from eventlens.review_queue import (
    convert_reviews_to_feedback,
    load_human_review_jsonl,
    load_review_queue_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    args = parser.parse_args()

    settings = load_settings()
    output = Path(args.output or settings.paths.feedback_store)
    schema = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    valid_event_types = {row.event_name for row in schema.definitions if row.scope == args.scope}
    queue = load_review_queue_jsonl(args.queue)
    reviews = load_human_review_jsonl(args.reviews)
    feedback = convert_reviews_to_feedback(
        queue,
        reviews,
        valid_event_types=valid_event_types,
    )

    existing = load_feedback_jsonl(output) if output.exists() else []
    existing_ids = {row.feedback_id for row in existing}
    new_rows = [row for row in feedback if row.feedback_id not in existing_ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        for row in new_rows:
            file.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "review_count": len(reviews),
                "approved_feedback_count": len(feedback),
                "appended_count": len(new_rows),
                "duplicate_feedback_count": len(feedback) - len(new_rows),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
