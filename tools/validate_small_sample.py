from __future__ import annotations

import argparse
from pathlib import Path

from eventlens.config import load_settings
from eventlens.event_retrieval import (
    EventSchemaIndex,
    OllamaEmbeddingClient,
    SubjectConstrainedEventRetriever,
    evaluate_recall_results,
)
from eventlens.io import read_articles_excel, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="artifacts/retrieval_sample")
    args = parser.parse_args()

    settings = load_settings()
    sheet_name = "个股新闻" if args.scope == "company" else "行业新闻"
    articles = read_articles_excel(settings.paths.tagged_train, sheet_name=sheet_name)
    index = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema if args.scope == "company" else None,
        industry_path=settings.paths.industry_event_schema if args.scope == "industry" else None,
    )
    eligible = [article for article in articles if index.candidates_for(article)]
    multi_candidate = [article for article in eligible if len(index.candidates_for(article)) > 1]
    single_candidate = [article for article in eligible if len(index.candidates_for(article)) == 1]
    sample = [*multi_candidate, *single_candidate][: args.limit]

    embedding = settings.event_retrieval.embedding
    retriever = SubjectConstrainedEventRetriever(
        index,
        OllamaEmbeddingClient(
            base_url=embedding.base_url,
            model=embedding.model,
            timeout_seconds=embedding.timeout_seconds,
            batch_size=embedding.batch_size,
        ),
        max_query_chars=settings.event_retrieval.max_query_chars,
    )
    results = retriever.recall_many(sample, top_k=args.top_k)
    metrics = evaluate_recall_results(results, top_k=args.top_k)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / f"{args.scope}_recall_details.jsonl", results)
    write_json(output_dir / f"{args.scope}_recall_metrics.json", metrics.model_dump())
    print(metrics.model_dump_json())


if __name__ == "__main__":
    main()
