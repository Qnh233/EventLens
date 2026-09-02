from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


INCLUDE_DIRS = ["src", "tools", "tests", "scripts", "configs", "deploy", "docs", "reports"]
INCLUDE_FILES = [
    "LICENSE",
    "pyproject.toml",
    "requirements-deploy.txt",
    "requirements-gpu.txt",
    "environment.yml",
    "Dockerfile",
    ".dockerignore",
    ".env.example",
]
SCHEMA_FILES = ["data/raw/事件类型_标的.json", "data/raw/事件类型_行业.json"]
ENTRY_DOCUMENTS = {
    "README.md": "submission_readme.md",
    "部署指南.md": "deployment.md",
    "交付合规清单.md": "submission_compliance.md",
}
IGNORED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    "node_modules",
    "site-packages",
    "vendor",
    "dist",
    "build",
}
IGNORED_FILE_SUFFIXES = (".pyc", ".pyo", ".o", ".obj")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ignore(_, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name in IGNORED_DIR_NAMES
        or name.endswith(IGNORED_FILE_SUFFIXES)
        or name.endswith(".egg-info")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建初赛 500MB 内可运行成果包")
    parser.add_argument("--output-dir", default="submission")
    parser.add_argument("--max-mb", type=float, default=500.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = root / args.output_dir
    stage = output_dir / "EventLens_Runtime"
    output_dir.mkdir(parents=True, exist_ok=True)
    # `_verify*` 是历史解压验收目录，不属于正式提交资产。
    for stale in output_dir.glob("_verify*"):
        if stale.is_dir():
            shutil.rmtree(stale)
        else:
            stale.unlink()
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    for name in INCLUDE_DIRS:
        source = root / name
        if source.exists():
            shutil.copytree(source, stage / name, ignore=_ignore)
    for name in INCLUDE_FILES + SCHEMA_FILES:
        source = root / name
        if source.exists():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    # 交付入口使用专门的简明 README，并把部署/合规文档提升到运行包根目录。
    for destination_name, source_name in ENTRY_DOCUMENTS.items():
        source = root / "docs" / source_name
        if not source.exists():
            raise RuntimeError(f"缺少提交入口文档: {source}")
        shutil.copy2(source, stage / destination_name)
        copied_in_docs = stage / "docs" / source_name
        if copied_in_docs.exists():
            copied_in_docs.unlink()

    # 明确禁止提交真实密钥与大体积原始新闻。研究态 BGE/LLM 不属于默认运行依赖，
    # 因此也不会由打包脚本从缓存目录隐式带入。
    forbidden_data = list((stage / "data").rglob("*.xlsx")) if (stage / "data").exists() else []
    forbidden_compiled = [
        path
        for path in stage.rglob("*")
        if path.name in IGNORED_DIR_NAMES
        or path.name.endswith(IGNORED_FILE_SUFFIXES)
        or path.name.endswith(".egg-info")
    ]
    forbidden_secrets = list(stage.rglob(".env")) + list(stage.rglob("*.pem"))
    if forbidden_data or forbidden_compiled or forbidden_secrets:
        raise RuntimeError(
            "submission 包包含禁止文件: "
            f"data={forbidden_data[:3]}, compiled={forbidden_compiled[:3]}, "
            f"secrets={forbidden_secrets[:3]}"
        )

    required_delivery_paths = [
        stage / "README.md",
        stage / "部署指南.md",
        stage / "交付合规清单.md",
        stage / "src" / "eventlens",
        stage / "scripts" / "run_deploy_smoke.py",
        stage / "requirements-deploy.txt",
        stage / "Dockerfile",
    ]
    missing_delivery_paths = [str(path.relative_to(stage)) for path in required_delivery_paths if not path.exists()]
    if missing_delivery_paths:
        raise RuntimeError(f"submission 包缺少交付必需项: {missing_delivery_paths}")

    model_manifest_path = stage / "deploy" / "models" / "manifest.json"
    runtime_manifest_path = stage / "deploy" / "runtime_manifest.json"
    if not model_manifest_path.exists() or not runtime_manifest_path.exists():
        raise RuntimeError("submission 包缺少模型或运行时权重清单")
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    required_weights = [row for row in runtime_manifest["models"] if row["runtime_required"]]
    missing_weights = [row["path"] for row in required_weights if not (stage / row["path"]).exists()]
    if missing_weights:
        raise RuntimeError(f"submission 包缺少运行时权重: {missing_weights}")
    mismatched_weights = [
        row["path"]
        for row in required_weights
        if row.get("sha256") and _sha256(stage / row["path"]) != row["sha256"]
    ]
    if mismatched_weights:
        raise RuntimeError(f"submission 运行时权重哈希不一致: {mismatched_weights}")

    zip_path = output_dir / "EventLens_Runtime.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in stage.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir))
    size_mb = zip_path.stat().st_size / 1024 / 1024
    if size_mb > args.max_mb:
        raise RuntimeError(f"submission zip {size_mb:.2f}MB 超过 {args.max_mb:.2f}MB")
    manifest = {
        "archive": str(zip_path.relative_to(root)),
        "archive_size_mb": round(size_mb, 3),
        "archive_sha256": _sha256(zip_path),
        "limit_mb": args.max_mb,
        "contains_local_cpu_models": (stage / "deploy/models/company/baseline.joblib").exists()
        and (stage / "deploy/models/industry/baseline.joblib").exists(),
        "runtime_required_model_count": len(required_weights),
        "runtime_model_download_required": bool(runtime_manifest["runtime_model_download_required"]),
        "experiment_report_count": len(
            [path for path in (stage / "reports").rglob("*") if path.is_file()]
        ),
        "model_manifest_mode": model_manifest.get("mode"),
        "contains_raw_news_xlsx": bool(list((stage / "data").rglob("*.xlsx"))) if (stage / "data").exists() else False,
        "contains_env_secret_file": bool(list(stage.rglob(".env"))),
        "contains_compiled_intermediates": bool(forbidden_compiled),
        "contains_vendored_dependencies": any(
            path.name in {"node_modules", "site-packages", "vendor", ".venv"}
            for path in stage.rglob("*")
        ),
        "database_required": False,
        "contains_chinese_deployment_guide": (stage / "部署指南.md").is_file(),
        "contains_submission_compliance_checklist": (stage / "交付合规清单.md").is_file(),
        "source_file_count": len(list((stage / "src").rglob("*.py"))),
    }
    (output_dir / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
