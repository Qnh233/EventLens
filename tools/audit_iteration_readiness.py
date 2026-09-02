from __future__ import annotations

import argparse
import json
from pathlib import Path


def count_review_status(review_dir: Path) -> dict[str, int]:
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    if not review_dir.exists():
        return counts
    for path in review_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                status = str(json.loads(line).get("status", ""))
            except json.JSONDecodeError:
                continue
            if status in counts:
                counts[status] += 1
    return counts


def build_readiness(*, remote_reachable: bool, bge_cache_ready: bool, review_dir: Path) -> dict:
    reviews = count_review_status(review_dir)
    approved_gold_ready = reviews["approved"] > 0
    return {
        "scope": "company",
        "remote_gpu_path_reachable": remote_reachable,
        "local_bge_m3_cache_ready": bge_cache_ready,
        "review_status": reviews,
        "approved_gold_ready": approved_gold_ready,
        "allowed_next_actions": {
            "triplet_seed42": remote_reachable,
            "production_oof_stability_backfill": remote_reachable or bge_cache_ready,
            "approved_gold_refresh": approved_gold_ready,
        },
        "decision": "ready" if (remote_reachable or bge_cache_ready or approved_gold_ready) else "blocked",
        "external_touched": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-reachable", action="store_true")
    parser.add_argument("--bge-cache-ready", action="store_true")
    parser.add_argument("--review-dir", default="artifacts/remote_archive/review_queue")
    parser.add_argument("--output", default="reports/iteration_readiness_gate.json")
    args = parser.parse_args()
    payload = build_readiness(
        remote_reachable=args.remote_reachable,
        bge_cache_ready=args.bge_cache_ready,
        review_dir=Path(args.review_dir),
    )
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
