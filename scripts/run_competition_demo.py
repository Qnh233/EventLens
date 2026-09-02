from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run(root: Path, *args: str) -> None:
    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run([sys.executable, "-m", "eventlens.cli", *args], cwd=root, env=env, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="EventLens 赛题一键演示")
    parser.add_argument("--scope", choices=["company", "industry", "both"], default="both")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-root", default="artifacts/competition_demo")
    parser.add_argument("--input", default="data/raw/news_without_tags_test.xlsx")
    parser.add_argument("--embeddings-dir", default="artifacts/embeddings/untagged_test")
    parser.add_argument("--agent-shadow", action="store_true")
    parser.add_argument("--agent-max-samples", type=int, default=3)
    args = parser.parse_args()
    if args.limit <= 0:
        raise ValueError("limit 必须大于 0")
    if args.agent_max_samples <= 0:
        raise ValueError("agent-max-samples 必须大于 0")

    root = Path(__file__).resolve().parents[1]
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scopes = ["company", "industry"] if args.scope == "both" else [args.scope]
    manifest: dict[str, object] = {
        "limit": args.limit,
        "agent_shadow": args.agent_shadow,
        "scopes": {},
    }

    for scope in scopes:
        out = output_root / scope
        predict_args = [
            "predict-assets",
            "--input",
            args.input,
            "--sheet-name",
            "0",
            "--embeddings-dir",
            args.embeddings_dir,
            "--routes",
            f"artifacts/subject_routes/{scope}_test.jsonl",
            "--recalls",
            f"artifacts/event_recall/{scope}_test.jsonl",
            "--scope",
            scope,
            "--model-dir",
            f"artifacts/models/{scope}",
            "--output-dir",
            str(out.relative_to(root)),
            "--limit",
            str(args.limit),
        ]
        if args.agent_shadow:
            predict_args.extend(["--agent-shadow", "--agent-max-samples", str(args.agent_max_samples)])
        _run(root, *predict_args)

        validation = output_root / f"{scope}_validation.json"
        _run(
            root,
            "validate-run",
            "--input-dir",
            str(out.relative_to(root)),
            "--output",
            str(validation.relative_to(root)),
        )
        scope_manifest: dict[str, object] = {
            "run_summary": _load_json(out / "run_summary.json"),
            "validation": _load_json(validation),
        }
        shadow_summary = out / "agent_shadow_summary.json"
        if shadow_summary.exists():
            scope_manifest["agent_shadow"] = _load_json(shadow_summary)
        manifest["scopes"][scope] = scope_manifest

    control_path = output_root / "control_safety.json"
    trust_path = output_root / "trust_control.json"
    _run(root, "benchmark-control-safety", "--output", str(control_path.relative_to(root)))
    _run(root, "benchmark-trust-controls", "--output", str(trust_path.relative_to(root)))
    manifest["control_safety"] = _load_json(control_path)
    manifest["trust_control"] = _load_json(trust_path)

    reference_scope = scopes[0]
    runtime_path = output_root / "runtime_plan.json"
    collection_path = output_root / "collection_requests.jsonl"
    _run(
        root,
        "runtime-plan",
        "--input-dir",
        str((output_root / reference_scope).relative_to(root)),
        "--queue-depth",
        "800",
        "--active-workers",
        "1",
        "--output",
        str(runtime_path.relative_to(root)),
        "--collection-output",
        str(collection_path.relative_to(root)),
    )
    manifest["runtime_plan"] = _load_json(runtime_path)

    manifest_path = output_root / "demo_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
