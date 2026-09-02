from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eventlens.webapp import AnalyzeRequest, WebArticle, analyze_payload


def main() -> None:
    samples = json.loads((ROOT / "deploy/sample_articles.json").read_text(encoding="utf-8"))
    payload = AnalyzeRequest(
        scope="company",
        articles=[WebArticle(**row) for row in samples[:3]],
    )
    result = analyze_payload(payload)
    assert result["summary"]["article_count"] == 3
    assert result["articles"]
    assert result["clusters"]
    assert result["alerts"]
    print(json.dumps({"passed": True, "summary": result["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
