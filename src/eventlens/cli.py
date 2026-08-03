from __future__ import annotations

import argparse
from pathlib import Path

from eventlens.baseline import TrainedBaseline, train_baseline
from eventlens.config import load_settings
from eventlens.evaluation import write_generalization_report
from eventlens.io import profile_articles, read_articles_excel, render_profile_markdown
from eventlens.pipeline import run_pipeline, write_pipeline_outputs


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(prog="eventlens")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--input", required=True)
    profile_parser.add_argument("--output", default=settings.paths.data_profile_report)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--model-dir", default=settings.paths.model_dir)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output-dir", default=settings.paths.output_dir)
    predict_parser.add_argument("--model-dir")

    gen_parser = subparsers.add_parser("generalization-report")
    gen_parser.add_argument("--input", required=True)
    gen_parser.add_argument("--output", default=settings.paths.generalization_report)

    args = parser.parse_args()
    if args.command == "profile":
        articles = read_articles_excel(args.input)
        markdown = render_profile_markdown(profile_articles(articles))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown, encoding="utf-8")
        return

    if args.command == "train":
        articles = read_articles_excel(args.input)
        model = train_baseline(articles, settings.model.model_dump())
        model.save(args.model_dir)
        return

    if args.command == "predict":
        articles = read_articles_excel(args.input)
        model = TrainedBaseline.load(args.model_dir) if args.model_dir else None
        result = run_pipeline(articles, model=model)
        write_pipeline_outputs(args.output_dir, result)
        return

    if args.command == "generalization-report":
        articles = read_articles_excel(args.input)
        write_generalization_report(args.output, articles)


if __name__ == "__main__":
    main()
