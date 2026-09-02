from __future__ import annotations

import argparse
import json

from eventlens.agent_evaluation import select_agent_case_indices
from eventlens.baseline import train_baseline
from eventlens.config import load_settings
from eventlens.embedding_export import load_exported_vectors
from eventlens.event_retrieval import (
    EventSchemaIndex,
    NativeSentenceTransformerEmbeddingClient,
    SubjectConstrainedEventRetriever,
)
from eventlens.io import read_competition_labeled_excel
from eventlens.subject_routing import SubjectRouter, SubjectRoutingPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["company", "industry"], default="company")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    settings = load_settings()
    train_articles = read_competition_labeled_excel(settings.paths.tagged_train)[
        f"{args.scope}_event"
    ]
    test_articles = read_competition_labeled_excel(settings.paths.tagged_test)[
        f"{args.scope}_event"
    ]
    _, vectors = load_exported_vectors(args.embeddings_dir)
    if len(vectors) != len(test_articles):
        raise ValueError("外部 embedding 数量与 labeled test 不一致")

    native = settings.native_embedding
    embedding_client = NativeSentenceTransformerEmbeddingClient(
        model=native.model,
        device=native.device,
        batch_size=native.batch_size,
        normalize_embeddings=native.normalize_embeddings,
        cache_folder=native.cache_folder,
        local_files_only=True,
    )
    schema_index = EventSchemaIndex.from_files(
        company_path=settings.paths.company_event_schema,
        industry_path=settings.paths.industry_event_schema,
    )
    router = SubjectRouter(
        schema_index,
        embedding_client,
        max_query_chars=settings.subject_routing.max_query_chars,
        min_alias_chars=settings.subject_routing.min_alias_chars,
    )
    routing_config = getattr(settings.subject_routing, args.scope)
    routes = router.route_from_vectors(
        test_articles,
        vectors,
        scope=args.scope,
        policy=SubjectRoutingPolicy(**routing_config.model_dump()),
    )
    subject_codes = [
        [route.accepted_subject_code]
        if route.accepted_subject_code
        else [candidate.subject_code for candidate in route.candidates]
        for route in routes
    ]
    recalls = SubjectConstrainedEventRetriever(
        schema_index,
        embedding_client,
        max_query_chars=settings.event_retrieval.max_query_chars,
    ).recall_from_vectors(
        [article.article_id for article in test_articles],
        vectors,
        subject_codes,
        scope=args.scope,
        top_k=3,
    )

    event_only_train = [
        article.model_copy(update={"polarity_label": None}) for article in train_articles
    ]
    baseline = train_baseline(event_only_train, settings.model.model_dump())
    predictions = [baseline.predict_one(article) for article in test_articles]
    ranked_indices = select_agent_case_indices(
        predictions,
        routes,
        recalls,
        max_samples=len(test_articles),
        confidence_max=settings.agent_expert.trigger_confidence_max,
        subject_margin_max=settings.agent_expert.trigger_subject_margin_max,
    )
    error_indices = {
        index
        for index, (article, prediction) in enumerate(zip(test_articles, predictions))
        if str(article.event_label) != prediction.event_type
    }
    selected = set(ranked_indices)
    selected_errors = selected & error_indices

    top_windows = {}
    for size in (30, 50, 100, 150, 200):
        top = set(ranked_indices[:size])
        errors = top & error_indices
        top_windows[str(size)] = {
            "selected_count": len(top),
            "error_count": len(errors),
            "error_rate": round(len(errors) / max(1, len(top)), 6),
            "error_coverage": round(len(errors) / max(1, len(error_indices)), 6),
        }

    payload = {
        "scope": args.scope,
        "sample_count": len(test_articles),
        "baseline_error_count": len(error_indices),
        "baseline_accuracy": round(
            (len(test_articles) - len(error_indices)) / max(1, len(test_articles)), 6
        ),
        "hard_route_count": sum(route.accepted_subject_code is not None for route in routes),
        "selector_count": len(selected),
        "selector_error_count": len(selected_errors),
        "selector_error_rate": round(len(selected_errors) / max(1, len(selected)), 6),
        "selector_error_coverage": round(
            len(selected_errors) / max(1, len(error_indices)), 6
        ),
        "top_windows": top_windows,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
