from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventlens.io import read_competition_labeled_excel
from eventlens.review_queue import build_human_review_packet, load_review_queue_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=76)
    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="可选逗号分隔 tranche，例如 76,76,77；总和必须等于队列长度",
    )
    parser.add_argument("--content-chars", type=int, default=1200)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    queue = load_review_queue_jsonl(args.queue)
    articles = read_competition_labeled_excel(args.input).get("company_event", [])
    packet = build_human_review_packet(queue, articles, content_chars=args.content_chars)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_sizes:
        batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(",") if value.strip()]
        if any(value <= 0 for value in batch_sizes):
            raise ValueError("batch-sizes must all be positive")
        if sum(batch_sizes) != len(packet):
            raise ValueError("batch-sizes sum must equal queue length")
    else:
        batch_sizes = [args.batch_size] * (len(packet) // args.batch_size)
        remainder = len(packet) % args.batch_size
        if remainder:
            batch_sizes.append(remainder)

    files: list[str] = []
    start = 0
    for batch_index, batch_size in enumerate(batch_sizes, start=1):
        batch = packet[start : start + batch_size]
        path = output_dir / f"batch_{batch_index:02d}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for row in batch:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        files.append(str(path))
        start += batch_size

    manifest = {
        "queue": args.queue,
        "article_count": len(packet),
        "batch_size": args.batch_size if not args.batch_sizes else None,
        "batch_sizes": batch_sizes,
        "batch_count": len(files),
        "content_chars": args.content_chars,
        "files": files,
        "gold_fields_excluded": ["event_label", "entity", "trading_code", "industry", "industry_code"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
