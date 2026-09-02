from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_EXTERNAL = 0.772131
EXPECTED_OOF = 0.772805
MAX_UNSAFE_CONTINUE_RATE = 0.0


def build_release_evidence(reports_dir: Path) -> dict:
    fusion_path = reports_dir / "complementary_prototype_fusion_company.json"
    goal_path = reports_dir / "goal_mode_complete.json"
    if not fusion_path.exists():
        raise FileNotFoundError(fusion_path)
    if not goal_path.exists():
        raise FileNotFoundError(goal_path)

    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    goal = json.loads(goal_path.read_text(encoding="utf-8"))
    external = float(fusion.get("external", {}).get("macro_f1", -1.0))
    oof = float(fusion.get("selected_oof", {}).get("macro_f1", -1.0))
    unsafe = float(
        goal.get("trust_and_runtime_controls", {}).get(
            "runtime_control_unsafe_continue_rate", 1.0
        )
    )
    bootstrap = fusion.get("oof_gain_bootstrap")
    folds = fusion.get("fold_stability")
    all_folds_positive = fusion.get("all_folds_positive")

    checks = {
        "production_like_external_matches_frozen_best": abs(external - EXPECTED_EXTERNAL) < 1e-9,
        "production_like_oof_matches_frozen_candidate": abs(oof - EXPECTED_OOF) < 1e-9,
        "unsafe_continue_rate_is_zero": unsafe <= MAX_UNSAFE_CONTINUE_RATE,
        "paired_oof_bootstrap_present": isinstance(bootstrap, dict) and bool(bootstrap),
        "fold_stability_present": isinstance(folds, list) and len(folds) == 3,
        "all_folds_positive": all_folds_positive is True,
    }
    missing = [name for name, passed in checks.items() if not passed]
    return {
        "scope": "company",
        "decision": "release_evidence_pass" if not missing else "conditional_hold",
        "strict_production_like_external_macro_f1": external,
        "duplication_safe_oof_macro_f1": oof,
        "unsafe_continue_rate": unsafe,
        "checks": checks,
        "missing_or_failed_evidence": missing,
        "external_touched_by_audit": False,
        "note": (
            "This audit only reads frozen reports. It never reads tagged external data or "
            "changes model, threshold, fusion, candidate or routing parameters."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--output",
        default="reports/competition_release_evidence_gate.json",
    )
    args = parser.parse_args()
    payload = build_release_evidence(Path(args.reports_dir))
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
