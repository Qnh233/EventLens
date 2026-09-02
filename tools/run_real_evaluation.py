from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from eventlens.evaluation import render_generalization_report
from eventlens.io import read_articles_excel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sheet-name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        articles = read_articles_excel(args.input, sheet_name=args.sheet_name)
        output.write_text(render_generalization_report(articles), encoding="utf-8")
    except Exception:
        error_path = output.with_suffix(".error.txt")
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(error_path.as_posix())
        raise
    print(output.as_posix())


if __name__ == "__main__":
    main()
