from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


SSH_OPTIONS = [
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=10",
    "-o",
    "ServerAliveCountMax=2",
]


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def ssh(remote: str, command: str, *, check: bool = True) -> str:
    result = run(["ssh", *SSH_OPTIONS, remote, command], check=check)
    return result.stdout.strip()


def remote_directory_exists(remote: str, path: str) -> bool:
    result = run(["ssh", *SSH_OPTIONS, remote, f"test -d {path!r}"], check=False)
    return result.returncode == 0


def copy_remote_directory(
    remote: str,
    remote_path: str,
    local_path: Path,
    *,
    attempts: int = 3,
) -> None:
    local_path.mkdir(parents=True, exist_ok=True)
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            run(["scp", *SSH_OPTIONS, "-r", f"{remote}:{remote_path}/.", str(local_path)])
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 * attempt)
    assert last_error is not None
    raise last_error


def directory_summary(path: Path) -> dict[str, int]:
    files = [file for file in path.rglob("*") if file.is_file()]
    return {
        "file_count": len(files),
        "bytes": sum(file.stat().st_size for file in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync EventLens remote reproducibility assets")
    parser.add_argument("--remote", default="eventlens-gpu")
    parser.add_argument("--remote-root", default="/root/autodl-tmp/EventLens")
    parser.add_argument("--destination", default="artifacts/remote_archive")
    parser.add_argument(
        "--include-bge",
        action="store_true",
        help="also copy the research-only BGE-M3 HF cache; this may be multiple GB",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    destination = root / args.destination
    destination.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] SSH preflight: {args.remote}")
    result = run(
        [
            "ssh",
            *SSH_OPTIONS,
            args.remote,
            f"test -d {args.remote_root!r} && echo EVENTLENS_REMOTE_OK",
        ],
        check=False,
    )
    if result.returncode != 0 or "EVENTLENS_REMOTE_OK" not in result.stdout:
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(
            "Remote EventLens is unavailable. Start the cloud instance and verify the SSH alias. "
            f"Details: {detail}"
        )

    print("[2/6] Save remote inventory")
    inventory_commands = [
        ("timestamp", "date -Is"),
        (
            "git",
            f"cd {args.remote_root!r}; git rev-parse HEAD 2>/dev/null; git status --short 2>/dev/null",
        ),
        (
            "reports",
            f"cd {args.remote_root!r}; du -sh reports 2>/dev/null; "
            "find reports -type f -printf '%s %p\\n' 2>/dev/null | sort -nr",
        ),
        (
            "logs",
            f"cd {args.remote_root!r}; du -sh artifacts/logs 2>/dev/null; "
            "find artifacts/logs -type f -printf '%s %p\\n' 2>/dev/null | sort -nr",
        ),
        (
            "embeddings",
            f"cd {args.remote_root!r}; du -h --max-depth=2 artifacts/embeddings 2>/dev/null | sort -h",
        ),
        (
            "review_assets",
            f"cd {args.remote_root!r}; du -h --max-depth=2 "
            "artifacts/review_packet artifacts/review_queue 2>/dev/null | sort -h",
        ),
        (
            "custom_model_files",
            f"cd {args.remote_root!r}; find . -type f "
            "\\( -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' "
            "-o -name '*.joblib' -o -name '*.onnx' -o -name '*.ckpt' \\) "
            "-printf '%s %p\\n' 2>/dev/null | sort -nr",
        ),
        (
            "bge_cache",
            "du -sh /root/autodl-tmp/hf_cache/hub/models--BAAI--bge-m3 2>/dev/null",
        ),
    ]
    inventory_parts: list[str] = []
    for name, command in inventory_commands:
        inventory_parts.append(f"=== {name} ===")
        inventory_parts.append(ssh(args.remote, command, check=False))
    (destination / "remote_inventory.txt").write_text(
        "\n".join(inventory_parts).rstrip() + "\n", encoding="utf-8"
    )

    print("[3/6] Sync reports and logs")
    copy_remote_directory(args.remote, f"{args.remote_root}/reports", destination / "reports")
    logs_path = f"{args.remote_root}/artifacts/logs"
    if remote_directory_exists(args.remote, logs_path):
        copy_remote_directory(args.remote, logs_path, destination / "logs")

    print("[4/6] Sync compact reproducibility assets")
    # 默认只同步核心 event/duplicate embeddings（约几十 MB），不把 1.2GB
    # untagged_train/test embedding 无差别拉回。后者不是部署运行依赖。
    embedding_names = [
        "company_event_train",
        "company_event_external",
        "company_duplicate_train",
        "company_duplicate_test",
        "industry_event_train",
        "industry_event_external",
        "industry_duplicate_train",
        "industry_duplicate_test",
    ]
    optional_directories = {
        **{
            f"embeddings/{name}": f"{args.remote_root}/artifacts/embeddings/{name}"
            for name in embedding_names
        },
        "review_packet": f"{args.remote_root}/artifacts/review_packet",
        "review_queue": f"{args.remote_root}/artifacts/review_queue",
    }
    for local_name, remote_path in optional_directories.items():
        if remote_directory_exists(args.remote, remote_path):
            try:
                copy_remote_directory(args.remote, remote_path, destination / local_name)
            except subprocess.CalledProcessError as exc:
                print(f"WARN: optional sync failed after retries: {remote_path}: {exc}")

    print("[5/6] Archive remote code snapshot for comparison")
    remote_archive = "/tmp/eventlens_repro_code.tar.gz"
    archive_command = (
        f"cd {args.remote_root!r}; tar -czf {remote_archive!r} "
        "src tools tests scripts configs pyproject.toml requirements*.txt environment.yml 2>/dev/null"
    )
    archive_result = run(["ssh", *SSH_OPTIONS, args.remote, archive_command], check=False)
    if archive_result.returncode == 0:
        run(
            [
                "scp",
                *SSH_OPTIONS,
                f"{args.remote}:{remote_archive}",
                str(destination / "remote_code_snapshot.tar.gz"),
            ]
        )
        run(["ssh", *SSH_OPTIONS, args.remote, f"rm -f {remote_archive!r}"], check=False)

    if args.include_bge:
        print("[6/6] Sync BGE-M3 research-only cache")
        bge_path = destination / "bge-m3"
        if bge_path.exists():
            shutil.rmtree(bge_path)
        run(
            [
                "scp",
                *SSH_OPTIONS,
                "-r",
                f"{args.remote}:/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-m3",
                str(bge_path),
            ]
        )
    else:
        print("[6/6] BGE-M3 skipped; pass --include-bge for the research archive")

    summaries = {
        path.name: directory_summary(path)
        for path in destination.iterdir()
        if path.is_dir()
    }
    manifest = {
        "remote": args.remote,
        "remote_root": args.remote_root,
        "include_bge": args.include_bge,
        "directories": summaries,
    }
    (destination / "sync_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Synced remote reproducibility assets to: {destination}")


if __name__ == "__main__":
    main()
