from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model_downloads_cn.json"
ALIYUN_PYPI = "https://mirrors.aliyun.com/pypi/simple/"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def load_catalog(path: Path = DEFAULT_CONFIG) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["models"]


def resolve_models(requested: list[str], catalog: dict[str, dict]) -> list[tuple[str, dict]]:
    names = list(catalog) if requested == ["all"] else requested
    unknown = sorted(set(names) - set(catalog))
    if unknown:
        raise ValueError(f"未知模型别名: {unknown}; 可选: {sorted(catalog)} 或 all")
    return [(name, catalog[name]) for name in names]


def _download_modelscope(model_id: str, target: Path) -> str:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "缺少 modelscope。请先执行: "
            f"python -m pip install -i {ALIYUN_PYPI} modelscope"
        ) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    return str(
        snapshot_download(
            model_id,
            local_dir=str(target),
        )
    )


def _download_hf_mirror(model_id: str, target: Path) -> str:
    # huggingface_hub 会在导入阶段读取部分 endpoint 配置，因此先设置环境变量，
    # 同时在 snapshot_download 中显式传 endpoint，避免进程里已有 HF 导入导致失效。
    os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "缺少 huggingface_hub。请先执行: "
            f"python -m pip install -i {ALIYUN_PYPI} huggingface_hub"
        ) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    return str(
        snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
            endpoint=HF_MIRROR_ENDPOINT,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="通过国内镜像下载 EventLens 可选研究/增强模型。"
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="模型别名，例如 bge-m3、bge-reranker-v2-m3 或 all",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "hf-mirror"],
        default="modelscope",
        help="默认 ModelScope 国内站；hf-mirror 作为备用。",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="模型目标目录中的相对路径以此目录为根。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印下载计划，不实际下载大文件。",
    )
    args = parser.parse_args()

    catalog = load_catalog()
    selected = resolve_models(args.models, catalog)
    plan: list[dict] = []
    for alias, item in selected:
        model_id = item[f"{args.source.replace('-', '_')}_id"] if args.source == "modelscope" else item["huggingface_id"]
        target = (args.root / item["target_dir"]).resolve()
        plan.append(
            {
                "alias": alias,
                "source": args.source,
                "model_id": model_id,
                "target": str(target),
                "approx_size": item["approx_size"],
                "runtime_required": item["runtime_required"],
            }
        )

    print(json.dumps({"download_plan": plan}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    downloaded: list[dict] = []
    for row in plan:
        target = Path(row["target"])
        if args.source == "modelscope":
            local_path = _download_modelscope(row["model_id"], target)
        else:
            local_path = _download_hf_mirror(row["model_id"], target)
        downloaded.append({**row, "local_path": local_path})
    print(json.dumps({"downloaded": downloaded}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
