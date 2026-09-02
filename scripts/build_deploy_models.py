from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eventlens.baseline import train_baseline
from eventlens.config import load_settings
from eventlens.io import read_competition_labeled_excel


def _no_subject(article):
    return article.model_copy(
        update={
            "entity": "",
            "industry": "",
            "trading_code": "",
            "industry_code": "",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 500MB 提交包使用的 CPU Lite 模型")
    parser.add_argument("--input", default="data/raw/news_with_tags_train.xlsx")
    parser.add_argument("--output-root", default="deploy/models")
    args = parser.parse_args()

    settings = load_settings()
    tasks = read_competition_labeled_excel(args.input)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "mode": "cpu_lite_no_labeled_subject_fields",
        "config_version": settings.app.version,
        "models": {},
    }
    for scope, key in (("company", "company_event"), ("industry", "industry_event")):
        articles = [_no_subject(row) for row in tasks[key]]
        model = train_baseline(articles, settings.model.model_dump())
        destination = output_root / scope
        model.save(destination)
        path = destination / "baseline.joblib"
        manifest["models"][scope] = {
            "train_count": len(articles),
            "label_count": len({str(row.event_label) for row in articles}),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
