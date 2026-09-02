from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def summarize_reports(reports: list[dict]) -> dict:
    """汇总固定配置多 seed OOF；禁止混入 external 或配置漂移。"""
    if not reports:
        raise ValueError("at least one report is required")
    if any(bool(row.get("external_touched")) for row in reports):
        raise ValueError("external-touched report cannot enter train-only multi-seed summary")

    configs: list[dict] = []
    seeds: list[int] = []
    for row in reports:
        config = dict(row.get("fixed_config") or {})
        if "random_state" not in config:
            raise ValueError("random_state missing from fixed config")
        seeds.append(int(config.pop("random_state")))
        configs.append(config)
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("fixed config mismatch across seed reports")
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate random_state in seed reports")

    baseline = [float(row["baseline_group_safe_oof"]["macro_f1"]) for row in reports]
    triplet = [float(row["triplet_fusion_oof"]["macro_f1"]) for row in reports]
    gains = [float(row["oof_gain"]) for row in reports]
    challenge_blocks = [row.get("challenge_slices") for row in reports]
    challenge_summary: dict[str, dict] | None = None
    stable_positive_slices: list[str] = []
    unstable_or_harmed_slices: list[str] = []
    if any(block is not None for block in challenge_blocks):
        if not all(isinstance(block, dict) for block in challenge_blocks):
            raise ValueError("challenge_slices must be present for every seed report")
        gain_maps = [dict(block.get("macro_f1_gain") or {}) for block in challenge_blocks]
        slice_names = list(gain_maps[0])
        if any(set(gain_map) != set(slice_names) for gain_map in gain_maps[1:]):
            raise ValueError("challenge slice mismatch across seed reports")
        challenge_summary = {}
        for name in slice_names:
            values = [float(gain_map[name]) for gain_map in gain_maps]
            sample_counts = [
                int(block["baseline"][name]["sample_count"])
                for block in challenge_blocks
            ]
            if len(set(sample_counts)) != 1:
                raise ValueError(f"challenge sample_count mismatch for {name}")
            challenge_summary[name] = {
                "sample_count": sample_counts[0],
                "macro_f1_gain": _stats(values),
                "positive_seed_count": sum(value > 0 for value in values),
                "nonnegative_all_seeds": all(value >= 0 for value in values),
            }
            if sample_counts[0] > 0 and all(value > 0 for value in values):
                stable_positive_slices.append(name)
            elif sample_counts[0] > 0:
                unstable_or_harmed_slices.append(name)
    order = sorted(range(len(seeds)), key=seeds.__getitem__)
    summary = {
        "protocol": "train-only fixed-config duplication-safe 3-fold OOF multi-seed stability",
        "seed_count": len(seeds),
        "seeds": [seeds[index] for index in order],
        "fixed_config_without_seed": configs[0],
        "all_external_untouched": True,
        "baseline_macro_f1": _stats(baseline),
        "triplet_macro_f1": _stats(triplet),
        "oof_gain": _stats(gains),
        "all_seed_gains_positive": all(value > 0 for value in gains),
        "all_seed_gains_pass_005": all(value >= 0.005 for value in gains),
    }
    if challenge_summary is not None:
        summary["challenge_slices"] = challenge_summary
        summary["challenge_stability_decision"] = {
            "stable_positive_slices": sorted(stable_positive_slices),
            "unstable_or_harmed_slices": sorted(unstable_or_harmed_slices),
            "selection_rule": "stable_positive requires >0 Macro-F1 gain for every fixed seed",
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    summary = summarize_reports(reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
