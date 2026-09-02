from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from eventlens.agent_evaluation import (
    evaluate_agent_decisions,
    run_agent_cases,
    select_agent_case_indices,
    summarize_agent_shadow,
    verify_agent_decisions,
)
from eventlens.asset_pipeline import run_asset_pipeline
from eventlens.baseline import TrainedBaseline, train_baseline
from eventlens.candidate_edges import (
    build_subject_time_edges,
    evaluate_duplicate_candidate_recall,
    load_candidate_edges_jsonl,
)
from eventlens.candidate_clustering import (
    evaluate_candidate_clusters,
    evaluate_candidate_rule_baseline,
)
from eventlens.cluster_benchmark import (
    build_cluster_benchmark_dataset,
    estimate_cluster_workload,
    run_cluster_grid_benchmark,
)
from eventlens.config import load_settings
from eventlens.control_safety_benchmark import benchmark_control_safety
from eventlens.duplicate_pairs import (
    build_duplicate_cluster_groups,
    build_duplicate_pairs,
    summarize_duplicate_pairs,
)
from eventlens.duplicate_pair_evaluation import (
    benchmark_duplicate_pairs,
    load_duplicate_pairs_jsonl,
    write_benchmark_report,
)
from eventlens.embedding_export import (
    export_article_embeddings,
    load_exported_article_ids,
    load_exported_vectors,
)
from eventlens.evaluation import write_generalization_report
from eventlens.event_external_evaluation import (
    evaluate_external_event_results,
    fit_embedding_linear_predictions,
)
from eventlens.event_retrieval import (
    EventSchemaIndex,
    NativeSentenceTransformerEmbeddingClient,
    OllamaEmbeddingClient,
    SubjectConstrainedEventRetriever,
    evaluate_recall_results,
)
from eventlens.hard_examples import build_hard_examples, load_routed_recalls_jsonl
from eventlens.io import (
    profile_articles,
    read_articles_excel,
    read_competition_labeled_excel,
    render_profile_markdown,
    write_json,
    write_jsonl,
)
from eventlens.learning import (
    FlywheelOrchestrator,
    SkillRegistry,
    load_feedback_jsonl,
    load_metrics_json,
)
from eventlens.llm_agent import (
    EventChangeVerifier,
    EventExpertAgent,
    OpenAICompatibleChatClient,
    TransformersChatClient,
)
from eventlens.pipeline import run_pipeline, write_pipeline_outputs
from eventlens.run_validation import validate_run_output_dir
from eventlens.runtime_control import (
    DependencyHealth,
    RuntimeController,
    RuntimeSnapshot,
)
from eventlens.evidence_control import EvidenceGateDecision
from eventlens.env import env_float, env_int, env_str, load_env_file
from eventlens.semantic_similarity import BgeSemanticPairScorer
from eventlens.subject_routing import (
    SubjectRouter,
    SubjectRoutingPolicy,
    load_subject_routes_jsonl,
    summarize_subject_routes,
)
from eventlens.trust_control_benchmark import benchmark_trust_controls
from eventlens.transformer_event import (
    TransformerTrainingConfig,
    classification_metrics as transformer_classification_metrics,
    run_transformer_event_experiment,
)


def main() -> None:
    load_env_file()
    settings = load_settings()
    parser = argparse.ArgumentParser(prog="eventlens")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--input", required=True)
    profile_parser.add_argument("--output", default=settings.paths.data_profile_report)
    profile_parser.add_argument("--sheet-name", default=0)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", required=True)
    train_parser.add_argument("--model-dir", default=settings.paths.model_dir)
    train_parser.add_argument("--sheet-name", default=0)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output-dir", default=settings.paths.output_dir)
    predict_parser.add_argument("--model-dir")
    predict_parser.add_argument(
        "--semantic-cluster",
        action="store_true",
        help="启用 Ollama BGE-M3 对规则近邻候选做语义复核",
    )

    asset_predict_parser = subparsers.add_parser("predict-assets")
    asset_predict_parser.add_argument("--input", default=settings.paths.untagged_test)
    asset_predict_parser.add_argument("--sheet-name", default=0)
    asset_predict_parser.add_argument("--embeddings-dir", required=True)
    asset_predict_parser.add_argument("--routes", required=True)
    asset_predict_parser.add_argument("--recalls", required=True)
    asset_predict_parser.add_argument(
        "--scope", choices=["company", "industry"], required=True
    )
    asset_predict_parser.add_argument("--model-dir", required=True)
    asset_predict_parser.add_argument("--output-dir", required=True)
    asset_predict_parser.add_argument("--limit", type=int, default=0)
    asset_predict_parser.add_argument(
        "--agent-shadow",
        action="store_true",
        help="仅对 hard-case 运行 LLM Expert+Verifier 并留痕，不覆盖正式预测",
    )
    asset_predict_parser.add_argument("--agent-max-samples", type=int, default=12)

    validate_run_parser = subparsers.add_parser("validate-run")
    validate_run_parser.add_argument("--input-dir", required=True)
    validate_run_parser.add_argument("--output")

    control_benchmark_parser = subparsers.add_parser("benchmark-control-safety")
    control_benchmark_parser.add_argument(
        "--output", default="reports/control_safety_benchmark.json"
    )

    trust_benchmark_parser = subparsers.add_parser("benchmark-trust-controls")
    trust_benchmark_parser.add_argument(
        "--output", default="reports/trust_control_benchmark.json"
    )

    runtime_plan_parser = subparsers.add_parser("runtime-plan")
    runtime_plan_parser.add_argument("--input-dir")
    runtime_plan_parser.add_argument("--queue-depth", type=int, default=0)
    runtime_plan_parser.add_argument("--active-workers", type=int, default=1)
    runtime_plan_parser.add_argument("--consecutive-failures", type=int, default=0)
    runtime_plan_parser.add_argument("--process-dead", action="store_true")
    runtime_plan_parser.add_argument(
        "--dependency",
        action="append",
        default=[],
        help="name:healthy|required_down|fallback",
    )
    runtime_plan_parser.add_argument("--output", default="artifacts/runtime/runtime_plan.json")
    runtime_plan_parser.add_argument(
        "--collection-output",
        default="artifacts/runtime/collection_requests.jsonl",
    )

    gen_parser = subparsers.add_parser("generalization-report")
    gen_parser.add_argument("--input", required=True)
    gen_parser.add_argument("--output", default=settings.paths.generalization_report)
    gen_parser.add_argument("--sheet-name", default=0)

    learning_parser = subparsers.add_parser("learning-cycle")
    learning_parser.add_argument("--feedback", default=settings.paths.feedback_store)
    learning_parser.add_argument("--baseline-metrics", required=True)
    learning_parser.add_argument("--candidate-metrics", required=True)
    learning_parser.add_argument("--registry", default=settings.paths.skill_registry)
    learning_parser.add_argument("--export-dir", default=settings.paths.skill_export_dir)
    learning_parser.add_argument("--human-approved", action="store_true")
    learning_parser.add_argument("--approved-by")

    pair_parser = subparsers.add_parser("build-duplicate-pairs")
    pair_parser.add_argument("--input", default=settings.paths.tagged_train)
    pair_parser.add_argument("--scope", choices=["company", "industry"], required=True)
    pair_parser.add_argument("--output")
    pair_parser.add_argument("--max-pairs", type=int, default=200)

    pair_eval_parser = subparsers.add_parser("evaluate-duplicate-pairs")
    pair_eval_parser.add_argument("--input", required=True)
    pair_eval_parser.add_argument("--output", required=True)

    recall_parser = subparsers.add_parser("recall-events")
    recall_parser.add_argument("--input", default=settings.paths.tagged_train)
    recall_parser.add_argument("--scope", choices=["company", "industry"], required=True)
    recall_parser.add_argument("--sheet-name")
    recall_parser.add_argument("--output-dir", default="artifacts/retrieval_sample")
    recall_parser.add_argument("--limit", type=int, default=settings.event_retrieval.sample_limit)
    recall_parser.add_argument("--top-k", type=int, default=settings.event_retrieval.top_k)

    cluster_benchmark_parser = subparsers.add_parser("benchmark-clusters")
    cluster_benchmark_parser.add_argument("--input", default=settings.paths.tagged_train)
    cluster_benchmark_parser.add_argument(
        "--scope", choices=["company", "industry"], required=True
    )
    cluster_benchmark_parser.add_argument("--output")
    cluster_benchmark_parser.add_argument(
        "--max-articles", type=int, default=settings.cluster_benchmark.max_articles
    )

    workload_parser = subparsers.add_parser("cluster-workload")
    workload_parser.add_argument("--input", default=settings.paths.untagged_train)
    workload_parser.add_argument("--sheet-name", default=0)
    workload_parser.add_argument(
        "--limit", type=int, default=settings.cluster_benchmark.stress_sample_size
    )
    workload_parser.add_argument(
        "--output", default="reports/cluster_workload_10k.json"
    )

    embedding_parser = subparsers.add_parser("encode-embeddings")
    embedding_parser.add_argument("--input", default=settings.paths.untagged_train)
    embedding_parser.add_argument("--sheet-name", default=0)
    embedding_parser.add_argument("--output-dir", required=True)
    embedding_parser.add_argument("--limit", type=int, default=0)

    route_parser = subparsers.add_parser("route-subjects")
    route_parser.add_argument("--input", default=settings.paths.untagged_train)
    route_parser.add_argument("--sheet-name", default=0)
    route_parser.add_argument("--embeddings-dir", required=True)
    route_parser.add_argument("--scope", choices=["company", "industry"], required=True)
    route_parser.add_argument("--output", required=True)

    routed_event_parser = subparsers.add_parser("recall-routed-events")
    routed_event_parser.add_argument("--embeddings-dir", required=True)
    routed_event_parser.add_argument("--routes", required=True)
    routed_event_parser.add_argument(
        "--scope", choices=["company", "industry"], required=True
    )
    routed_event_parser.add_argument("--output", required=True)
    routed_event_parser.add_argument("--top-k", type=int, default=3)

    edge_parser = subparsers.add_parser("build-candidate-edges")
    edge_parser.add_argument("--input", default=settings.paths.untagged_train)
    edge_parser.add_argument("--sheet-name", default=0)
    edge_parser.add_argument("--routes", required=True)
    edge_parser.add_argument("--output", required=True)

    edge_eval_parser = subparsers.add_parser("evaluate-candidate-edges")
    edge_eval_parser.add_argument("--input", default=settings.paths.tagged_train)
    edge_eval_parser.add_argument("--sheet-name", required=True)
    edge_eval_parser.add_argument("--edges", required=True)
    edge_eval_parser.add_argument("--output", required=True)

    candidate_cluster_parser = subparsers.add_parser("evaluate-candidate-clusters")
    candidate_cluster_parser.add_argument("--input", default=settings.paths.tagged_train)
    candidate_cluster_parser.add_argument("--sheet-name", required=True)
    candidate_cluster_parser.add_argument("--embeddings-dir", required=True)
    candidate_cluster_parser.add_argument("--edges", required=True)
    candidate_cluster_parser.add_argument("--recalls", required=True)
    candidate_cluster_parser.add_argument("--output", required=True)
    candidate_cluster_parser.add_argument("--decisions-output")
    candidate_cluster_parser.add_argument("--baseline-output")

    hard_example_parser = subparsers.add_parser("build-hard-examples")
    hard_example_parser.add_argument("--routes", required=True)
    hard_example_parser.add_argument("--recalls", required=True)
    hard_example_parser.add_argument("--output", required=True)

    external_event_parser = subparsers.add_parser("evaluate-event-external")
    external_event_parser.add_argument(
        "--scope", choices=["company", "industry"], required=True
    )
    external_event_parser.add_argument("--embeddings-dir", required=True)
    external_event_parser.add_argument("--train-embeddings-dir")
    external_event_parser.add_argument("--output", required=True)
    external_event_parser.add_argument("--top-k", type=int, default=3)

    transformer_event_parser = subparsers.add_parser("evaluate-transformer-event")
    transformer_event_parser.add_argument(
        "--scope", choices=["company", "industry"], required=True
    )
    transformer_event_parser.add_argument(
        "--model", default=settings.transformer_event.model
    )
    transformer_event_parser.add_argument("--output", required=True)
    transformer_event_parser.add_argument("--model-dir")
    transformer_event_parser.add_argument("--local-files-only", action="store_true")

    agent_eval_parser = subparsers.add_parser("evaluate-agent-expert")
    agent_eval_parser.add_argument(
        "--scope", choices=["company", "industry"], required=True
    )
    agent_eval_parser.add_argument("--embeddings-dir", required=True)
    agent_eval_parser.add_argument(
        "--provider",
        choices=["local_transformers", "openai_compatible"],
        default=env_str("EVENTLENS_LLM_PROVIDER", settings.agent_expert.provider),
    )
    agent_eval_parser.add_argument(
        "--model", default=env_str("EVENTLENS_LLM_MODEL", settings.agent_expert.model)
    )
    agent_eval_parser.add_argument(
        "--base-url",
        default=env_str("EVENTLENS_LLM_BASE_URL", settings.agent_expert.base_url),
    )
    agent_eval_parser.add_argument(
        "--api-key-env",
        default=env_str("EVENTLENS_LLM_API_KEY_ENV", settings.agent_expert.api_key_env),
    )
    agent_eval_parser.add_argument("--max-samples", type=int, default=12)
    agent_eval_parser.add_argument("--top-k", type=int, default=3)
    agent_eval_parser.add_argument("--local-files-only", action="store_true")
    agent_eval_parser.add_argument("--no-verifier", action="store_true")
    agent_eval_parser.add_argument("--output", required=True)
    agent_eval_parser.add_argument("--trace-output", required=True)

    args = parser.parse_args()
    if args.command == "profile":
        articles = read_articles_excel(args.input, sheet_name=args.sheet_name)
        markdown = render_profile_markdown(profile_articles(articles))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(markdown, encoding="utf-8")
        return

    if args.command == "train":
        articles = read_articles_excel(args.input, sheet_name=args.sheet_name)
        model = train_baseline(articles, settings.model.model_dump())
        model.save(args.model_dir)
        return

    if args.command == "predict":
        articles = read_articles_excel(args.input)
        model = TrainedBaseline.load(args.model_dir) if args.model_dir else None
        cluster_config = settings.cluster.model_dump()
        if args.semantic_cluster:
            cluster_config["semantic"]["enabled"] = True
        result = run_pipeline(
            articles,
            model=model,
            cluster_config=cluster_config,
            skill_registry_path=settings.paths.skill_registry,
        )
        write_pipeline_outputs(
            args.output_dir,
            result,
            lifecycle_ledger_path=settings.paths.lifecycle_ledger,
        )
        return

    if args.command == "predict-assets":
        manifest, vectors = load_exported_vectors(args.embeddings_dir)
        article_ids = load_exported_article_ids(args.embeddings_dir)
        count = manifest.article_count if args.limit <= 0 else min(args.limit, manifest.article_count)
        articles = read_articles_excel(
            args.input,
            sheet_name=args.sheet_name,
            nrows=count,
        )
        routes = load_subject_routes_jsonl(args.routes, limit=count)
        recalls = load_routed_recalls_jsonl(args.recalls, limit=count)
        expected_ids = article_ids[:count]
        if expected_ids != [row.article_id for row in articles]:
            raise ValueError("输入文章顺序与 embedding index 不一致")
        if expected_ids != [row.article_id for row in routes]:
            raise ValueError("主体路由顺序与 embedding index 不一致")
        if expected_ids != [row.article_id for row in recalls]:
            raise ValueError("事件召回顺序与 embedding index 不一致")
        edges = build_subject_time_edges(
            articles,
            routes,
            window_days=settings.cluster.time_window_days,
            max_neighbors_per_article=settings.cluster.top_k,
        )
        model = TrainedBaseline.load(args.model_dir)
        result = run_asset_pipeline(
            articles,
            vectors[:count],
            routes,
            recalls,
            edges,
            scope=args.scope,
            model=model,
            skill_registry_path=settings.paths.skill_registry,
        )
        output_dir = Path(args.output_dir)
        write_pipeline_outputs(output_dir, result, lifecycle_ledger_path=None)
        write_jsonl(output_dir / "subject_route.jsonl", routes)
        write_jsonl(output_dir / "event_recall.jsonl", recalls)
        write_jsonl(output_dir / "candidate_edge.jsonl", edges)
        agent_shadow_summary = None
        if args.agent_shadow:
            if args.agent_max_samples <= 0:
                raise ValueError("agent-max-samples 必须大于 0")
            provider = env_str("EVENTLENS_LLM_PROVIDER", settings.agent_expert.provider)
            if provider != "openai_compatible":
                raise ValueError("predict-assets 的 agent shadow 当前只允许外部 openai_compatible provider")
            chat_client = OpenAICompatibleChatClient(
                base_url=env_str("EVENTLENS_LLM_BASE_URL", settings.agent_expert.base_url),
                model=env_str("EVENTLENS_LLM_MODEL", settings.agent_expert.model),
                api_key_env=env_str(
                    "EVENTLENS_LLM_API_KEY_ENV", settings.agent_expert.api_key_env
                ),
                temperature=env_float(
                    "EVENTLENS_LLM_TEMPERATURE", settings.agent_expert.temperature
                ),
                max_tokens=env_int(
                    "EVENTLENS_LLM_MAX_TOKENS", settings.agent_expert.max_tokens
                ),
                timeout_seconds=env_float("EVENTLENS_LLM_TIMEOUT_SECONDS", 180.0),
                thinking=env_str("EVENTLENS_LLM_THINKING", "enabled"),
                reasoning_effort=env_str("EVENTLENS_LLM_REASONING_EFFORT", "max"),
            )
            indices = select_agent_case_indices(
                result["predictions"],
                routes,
                recalls,
                max_samples=args.agent_max_samples,
                confidence_max=env_float(
                    "EVENTLENS_AGENT_TRIGGER_CONFIDENCE_MAX",
                    settings.agent_expert.trigger_confidence_max,
                ),
                subject_margin_max=env_float(
                    "EVENTLENS_AGENT_TRIGGER_SUBJECT_MARGIN_MAX",
                    settings.agent_expert.trigger_subject_margin_max,
                ),
            )
            agent = EventExpertAgent(
                chat_client,
                max_steps=env_int("EVENTLENS_AGENT_MAX_STEPS", settings.agent_expert.max_steps),
                max_content_chars=env_int(
                    "EVENTLENS_AGENT_MAX_CONTENT_CHARS",
                    settings.agent_expert.max_content_chars,
                ),
            )
            decisions = run_agent_cases(
                agent,
                articles,
                result["predictions"],
                routes,
                recalls,
                indices,
            )
            verifier = EventChangeVerifier(
                chat_client,
                max_content_chars=env_int(
                    "EVENTLENS_AGENT_MAX_CONTENT_CHARS",
                    settings.agent_expert.max_content_chars,
                ),
            )
            decisions = verify_agent_decisions(
                verifier,
                articles,
                result["predictions"],
                recalls,
                indices,
                decisions,
            )
            agent_shadow_summary = summarize_agent_shadow(decisions).model_dump()
            agent_shadow_summary.update(
                {
                    "mode": "shadow_only",
                    "provider": provider,
                    "model": chat_client.model,
                    "selected_indices": indices,
                    "api_usage": chat_client.usage,
                }
            )
            write_jsonl(output_dir / "agent_shadow.jsonl", decisions)
            write_json(output_dir / "agent_shadow_summary.json", agent_shadow_summary)
        summary = {
            "scope": args.scope,
            "article_count": count,
            "event_count": sum(row.has_event for row in result["predictions"]),
            "cluster_count": len(result["clusters"]),
            "candidate_edge_count": len(edges),
            "merged_edge_count": sum(row.merged for row in result["cluster_decisions"]),
            "alert_count": len(result["alerts"]),
            "blocked_alert_count": sum(
                not row.delivery_allowed for row in result["alerts"]
            ),
            "claim_count": len(result.get("claim_bindings", [])),
            "evidence_gate_count": len(result.get("evidence_gates", [])),
            "lifecycle_count": len(result["lifecycles"]),
            "learning_signal_count": len(result["learning_signals"]),
            "agent_shadow_count": (
                agent_shadow_summary["sample_count"] if agent_shadow_summary else 0
            ),
        }
        write_json(output_dir / "run_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False))
        return

    if args.command == "validate-run":
        report = validate_run_output_dir(args.input_dir)
        if args.output:
            write_json(args.output, report.model_dump())
        print(report.model_dump_json())
        if not report.passed:
            raise SystemExit(2)
        return

    if args.command == "benchmark-control-safety":
        report = benchmark_control_safety(settings.runtime_control.model_dump())
        write_json(args.output, report.model_dump())
        print(report.model_dump_json())
        return

    if args.command == "benchmark-trust-controls":
        report = benchmark_trust_controls(settings.evidence_control.model_dump())
        write_json(args.output, report.model_dump())
        print(report.model_dump_json())
        return

    if args.command == "runtime-plan":
        run_valid = True
        gates: list[EvidenceGateDecision] = []
        if args.input_dir:
            root = Path(args.input_dir)
            try:
                run_valid = validate_run_output_dir(root).passed
            except (FileNotFoundError, ValueError):
                run_valid = False
            gate_path = root / "evidence_gate.jsonl"
            if gate_path.exists():
                gates = [
                    EvidenceGateDecision.model_validate_json(line)
                    for line in gate_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        dependencies = [_parse_dependency(value) for value in args.dependency]
        plan = RuntimeController(settings.runtime_control.model_dump()).plan(
            RuntimeSnapshot(
                queue_depth=args.queue_depth,
                active_workers=args.active_workers,
                consecutive_failures=args.consecutive_failures,
                process_alive=not args.process_dead,
                run_valid=run_valid,
                dependencies=dependencies,
            ),
            gates,
        )
        write_json(args.output, plan.model_dump())
        write_jsonl(args.collection_output, plan.collection_requests)
        print(plan.model_dump_json())
        return

    if args.command == "generalization-report":
        articles = read_articles_excel(args.input, sheet_name=args.sheet_name)
        write_generalization_report(args.output, articles)
        return

    if args.command == "learning-cycle":
        registry = SkillRegistry(args.registry)
        config = settings.learning
        orchestrator = FlywheelOrchestrator(
            registry=registry,
            min_feedback_count=config.min_feedback_count,
            min_macro_f1_gain=config.min_macro_f1_gain,
            max_critical_error_regression=config.max_critical_error_regression,
            require_human_approval=config.require_human_approval,
        )
        result = orchestrator.run(
            feedback=load_feedback_jsonl(args.feedback),
            baseline_metrics=load_metrics_json(args.baseline_metrics),
            candidate_metrics=load_metrics_json(args.candidate_metrics),
            human_approved=args.human_approved,
            approved_by=args.approved_by,
        )
        exported = registry.export_active_skills(args.export_dir)
        print(
            f"候选={len(result.candidates)}，激活={result.promoted_count}，"
            f"拒绝={result.rejected_count}，导出={len(exported)}"
        )
        return

    if args.command == "build-duplicate-pairs":
        datasets = read_competition_labeled_excel(args.input)
        config = settings.duplicate_pairs
        pairs = build_duplicate_pairs(
            datasets,
            scope=args.scope,
            negative_ratio=config.negative_ratio,
            max_positive_pairs_per_group=config.max_positive_pairs_per_group,
            time_window_days=config.time_window_days,
            max_text_chars=config.max_text_chars,
            seed=config.seed,
            max_pairs=args.max_pairs,
            subject_lead_chars=config.subject_lead_chars,
            min_subject_alias_chars=config.min_subject_alias_chars,
            require_resolved_subject_for_negatives=(
                config.require_resolved_subject_for_negatives
            ),
        )
        output = Path(args.output or f"artifacts/training_pairs/{args.scope}_pairs.jsonl")
        write_jsonl(output, pairs)
        summary = summarize_duplicate_pairs(pairs)
        write_json(output.with_suffix(".summary.json"), summary.model_dump())
        print(summary.model_dump_json())
        return

    if args.command == "evaluate-duplicate-pairs":
        embedding = settings.event_retrieval.embedding
        benchmark = benchmark_duplicate_pairs(
            load_duplicate_pairs_jsonl(args.input),
            OllamaEmbeddingClient(
                base_url=embedding.base_url,
                model=embedding.model,
                timeout_seconds=embedding.timeout_seconds,
                batch_size=embedding.batch_size,
                num_gpu=embedding.num_gpu,
            ),
            calibration_ratio=settings.duplicate_pair_evaluation.calibration_ratio,
            seed=settings.duplicate_pair_evaluation.seed,
        )
        write_benchmark_report(args.output, benchmark)
        print(benchmark.model_dump_json())
        return

    if args.command == "recall-events":
        sheet_name = args.sheet_name or ("个股新闻" if args.scope == "company" else "行业新闻")
        articles = read_articles_excel(args.input, sheet_name=sheet_name)
        schema_index = EventSchemaIndex.from_files(
            company_path=settings.paths.company_event_schema if args.scope == "company" else None,
            industry_path=settings.paths.industry_event_schema if args.scope == "industry" else None,
        )
        eligible = [article for article in articles if schema_index.candidates_for(article)]
        multi_candidate = [
            article for article in eligible if len(schema_index.candidates_for(article)) > 1
        ]
        single_candidate = [
            article for article in eligible if len(schema_index.candidates_for(article)) == 1
        ]
        sample = [*multi_candidate, *single_candidate][: max(0, args.limit)]
        embedding = settings.event_retrieval.embedding
        retriever = SubjectConstrainedEventRetriever(
            schema_index,
            OllamaEmbeddingClient(
                base_url=embedding.base_url,
                model=embedding.model,
                timeout_seconds=embedding.timeout_seconds,
                batch_size=embedding.batch_size,
                num_gpu=embedding.num_gpu,
            ),
            max_query_chars=settings.event_retrieval.max_query_chars,
        )
        details = retriever.recall_many(sample, top_k=args.top_k)
        evaluation = evaluate_recall_results(details, top_k=args.top_k)
        output_dir = Path(args.output_dir)
        write_jsonl(output_dir / f"{args.scope}_recall_details.jsonl", details)
        write_json(
            output_dir / f"{args.scope}_recall_metrics.json",
            evaluation.model_dump(),
        )
        print(evaluation.model_dump_json())
        return

    if args.command == "benchmark-clusters":
        datasets = read_competition_labeled_excel(args.input)
        pair_config = settings.duplicate_pairs
        groups = build_duplicate_cluster_groups(
            datasets,
            scope=args.scope,
            max_articles=args.max_articles,
            subject_lead_chars=pair_config.subject_lead_chars,
            min_subject_alias_chars=pair_config.min_subject_alias_chars,
        )
        dataset = build_cluster_benchmark_dataset(groups, scope=args.scope)
        semantic = settings.cluster.semantic
        embedding = semantic.embedding
        scorer = BgeSemanticPairScorer(
            OllamaEmbeddingClient(
                base_url=embedding.base_url,
                model=embedding.model,
                timeout_seconds=embedding.timeout_seconds,
                batch_size=embedding.batch_size,
            ),
            model=embedding.model,
            cache_path=semantic.cache_path,
        )
        benchmark = settings.cluster_benchmark
        report = run_cluster_grid_benchmark(
            dataset,
            cluster_config=settings.cluster.model_dump(),
            semantic_scorer=scorer,
            candidate_thresholds=benchmark.candidate_thresholds,
            semantic_thresholds=benchmark.semantic_thresholds,
            top_ks=benchmark.top_ks,
            minimum_b_cubed_f1_gain=benchmark.minimum_b_cubed_f1_gain,
            minimum_pairwise_recall_gain=benchmark.minimum_pairwise_recall_gain,
        )
        output = args.output or f"reports/cluster_benchmark_{args.scope}.json"
        write_json(output, report.model_dump())
        print(report.model_dump_json())
        return

    if args.command == "cluster-workload":
        articles = read_articles_excel(
            args.input,
            sheet_name=args.sheet_name,
            nrows=max(0, args.limit),
        )
        report = estimate_cluster_workload(
            articles,
            time_window_days=settings.cluster.time_window_days,
            top_k=max(settings.cluster_benchmark.top_ks),
        )
        write_json(args.output, report.model_dump())
        print(report.model_dump_json())
        return

    if args.command == "encode-embeddings":
        export_cfg = settings.embedding_export
        articles = read_articles_excel(
            args.input,
            sheet_name=args.sheet_name,
            nrows=args.limit if args.limit > 0 else None,
        )
        native = settings.native_embedding
        client = NativeSentenceTransformerEmbeddingClient(
            model=native.model,
            device=native.device,
            batch_size=native.batch_size,
            normalize_embeddings=native.normalize_embeddings,
            cache_folder=native.cache_folder,
            local_files_only=native.local_files_only,
        )
        manifest = export_article_embeddings(
            articles,
            client,
            output_dir=args.output_dir,
            model_id=native.model,
            max_content_chars=export_cfg.max_content_chars,
            chunk_size=export_cfg.chunk_size,
        )
        print(manifest.model_dump_json())
        return

    if args.command == "route-subjects":
        manifest, vectors = load_exported_vectors(args.embeddings_dir)
        articles = read_articles_excel(
            args.input,
            sheet_name=args.sheet_name,
            nrows=manifest.article_count,
        )
        index = EventSchemaIndex.from_files(
            company_path=settings.paths.company_event_schema,
            industry_path=settings.paths.industry_event_schema,
        )
        native = settings.native_embedding
        router = SubjectRouter(
            index,
            NativeSentenceTransformerEmbeddingClient(
                model=native.model,
                device=native.device,
                batch_size=native.batch_size,
                normalize_embeddings=native.normalize_embeddings,
                cache_folder=native.cache_folder,
                local_files_only=native.local_files_only,
            ),
            max_query_chars=settings.subject_routing.max_query_chars,
            min_alias_chars=settings.subject_routing.min_alias_chars,
        )
        policy_cfg = getattr(settings.subject_routing, args.scope)
        rows = router.route_from_vectors(
            articles,
            vectors,
            scope=args.scope,
            policy=SubjectRoutingPolicy(**policy_cfg.model_dump()),
        )
        output = Path(args.output)
        write_jsonl(output, rows)
        summary = summarize_subject_routes(rows, scope=args.scope)
        write_json(output.with_suffix(".summary.json"), summary.model_dump())
        print(summary.model_dump_json())
        return

    if args.command == "recall-routed-events":
        manifest, vectors = load_exported_vectors(args.embeddings_dir)
        embedding_article_ids = load_exported_article_ids(args.embeddings_dir)
        routes = load_subject_routes_jsonl(args.routes)
        if len(routes) != manifest.article_count:
            raise ValueError("主体路由数量与 embedding manifest 不一致")
        route_article_ids = [row.article_id for row in routes]
        if route_article_ids != embedding_article_ids:
            raise ValueError("主体路由顺序与 embedding index 不一致")
        index = EventSchemaIndex.from_files(
            company_path=settings.paths.company_event_schema,
            industry_path=settings.paths.industry_event_schema,
        )
        native = settings.native_embedding
        retriever = SubjectConstrainedEventRetriever(
            index,
            NativeSentenceTransformerEmbeddingClient(
                model=native.model,
                device=native.device,
                batch_size=native.batch_size,
                normalize_embeddings=native.normalize_embeddings,
                cache_folder=native.cache_folder,
                local_files_only=native.local_files_only,
            ),
            max_query_chars=settings.event_retrieval.max_query_chars,
        )
        subject_codes = [
            [row.accepted_subject_code]
            if row.accepted_subject_code
            else [candidate.subject_code for candidate in row.candidates]
            for row in routes
        ]
        results = retriever.recall_from_vectors(
            embedding_article_ids,
            vectors,
            subject_codes,
            scope=args.scope,
            top_k=args.top_k,
        )
        write_jsonl(args.output, results)
        print(
            json.dumps(
                {
                    "scope": args.scope,
                    "article_count": len(results),
                    "with_event_candidates": sum(bool(row.candidates) for row in results),
                    "top_k": args.top_k,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "build-candidate-edges":
        routes = load_subject_routes_jsonl(args.routes)
        articles = read_articles_excel(
            args.input,
            sheet_name=args.sheet_name,
            nrows=len(routes),
        )
        edges = build_subject_time_edges(
            articles,
            routes,
            window_days=settings.cluster.time_window_days,
            max_neighbors_per_article=settings.cluster.top_k,
        )
        write_jsonl(args.output, edges)
        print(
            json.dumps(
                {
                    "article_count": len(articles),
                    "edge_count": len(edges),
                    "time_window_days": settings.cluster.time_window_days,
                    "max_neighbors_per_article": settings.cluster.top_k,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "evaluate-candidate-edges":
        articles = read_articles_excel(args.input, sheet_name=args.sheet_name)
        edges = load_candidate_edges_jsonl(args.edges)
        cfg = settings.candidate_edge_evaluation
        report = evaluate_duplicate_candidate_recall(
            articles,
            edges,
            window_days=settings.cluster.time_window_days,
            minimum_eligible_recall=cfg.minimum_eligible_recall,
        )
        write_json(args.output, report.model_dump())
        print(report.model_dump_json())
        if not report.passed:
            raise SystemExit(2)
        return

    if args.command == "evaluate-candidate-clusters":
        articles = read_articles_excel(args.input, sheet_name=args.sheet_name)
        manifest, vectors = load_exported_vectors(args.embeddings_dir)
        article_ids = load_exported_article_ids(args.embeddings_dir)
        if manifest.article_count != len(articles) or article_ids != [
            row.article_id for row in articles
        ]:
            raise ValueError("embedding index 与评测文章顺序不一致")
        edges = load_candidate_edges_jsonl(args.edges)
        recalls = load_routed_recalls_jsonl(args.recalls)
        truth = {
            row.article_id: row.duplication_id
            for row in articles
            if row.duplication_id
        }
        if len(truth) != len(articles):
            raise ValueError("候选聚类评测要求每篇文章都有 duplication_id")
        report, decisions = evaluate_candidate_clusters(
            article_ids,
            vectors,
            edges,
            recalls,
            truth,
            similarity_threshold=settings.cluster.semantic.similarity_threshold,
        )
        write_json(args.output, report.model_dump())
        if args.baseline_output:
            baseline = evaluate_candidate_rule_baseline(
                articles,
                edges,
                recalls,
                truth,
                cluster_config=settings.cluster.model_dump(),
            )
            write_json(args.baseline_output, baseline.model_dump())
        if args.decisions_output:
            write_jsonl(args.decisions_output, decisions)
        print(report.model_dump_json())
        return

    if args.command == "build-hard-examples":
        routes = load_subject_routes_jsonl(args.routes)
        recalls = load_routed_recalls_jsonl(args.recalls)
        cfg = settings.hard_examples
        route_scope = routes[0].scope if routes else "company"
        routing_cfg = getattr(settings.subject_routing, route_scope)
        examples = build_hard_examples(
            routes,
            recalls,
            subject_margin_threshold=cfg.subject_margin_threshold,
            event_margin_threshold=cfg.event_margin_threshold,
            include_subject_rejection_signal=(
                routing_cfg.exact_alias_hard_route or routing_cfg.bge_hard_route
            ),
            max_examples=cfg.max_examples,
        )
        write_jsonl(args.output, examples)
        print(
            json.dumps(
                {
                    "article_count": len(routes),
                    "hard_example_count": len(examples),
                    "output": args.output,
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "evaluate-agent-expert":
        if args.max_samples <= 0:
            raise ValueError("max-samples 必须大于 0")
        train_sets = read_competition_labeled_excel(settings.paths.tagged_train)
        test_sets = read_competition_labeled_excel(settings.paths.tagged_test)
        key = f"{args.scope}_event"
        train_articles = train_sets[key]
        test_articles = test_sets[key]
        manifest, vectors = load_exported_vectors(args.embeddings_dir)
        article_ids = load_exported_article_ids(args.embeddings_dir)
        if manifest.article_count != len(test_articles) or article_ids != [
            row.article_id for row in test_articles
        ]:
            raise ValueError("Agent 评测 embedding 与 tagged test 顺序不一致")

        native = settings.native_embedding
        embedding_client = NativeSentenceTransformerEmbeddingClient(
            model=native.model,
            device=native.device,
            batch_size=native.batch_size,
            normalize_embeddings=native.normalize_embeddings,
            cache_folder=native.cache_folder,
            local_files_only=native.local_files_only,
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
        policy_cfg = getattr(settings.subject_routing, args.scope)
        routes = router.route_from_vectors(
            test_articles,
            vectors,
            scope=args.scope,
            policy=SubjectRoutingPolicy(**policy_cfg.model_dump()),
        )
        subject_codes = [
            [row.accepted_subject_code]
            if row.accepted_subject_code
            else [candidate.subject_code for candidate in row.candidates]
            for row in routes
        ]
        retriever = SubjectConstrainedEventRetriever(
            schema_index,
            embedding_client,
            max_query_chars=settings.event_retrieval.max_query_chars,
        )
        recalls = retriever.recall_from_vectors(
            article_ids,
            vectors,
            subject_codes,
            scope=args.scope,
            top_k=args.top_k,
        )
        event_only_train = [
            row.model_copy(update={"polarity_label": None}) for row in train_articles
        ]
        baseline_model = train_baseline(event_only_train, settings.model.model_dump())
        predictions = [baseline_model.predict_one(row) for row in test_articles]
        indices = select_agent_case_indices(
            predictions,
            routes,
            recalls,
            max_samples=args.max_samples,
            confidence_max=env_float(
                "EVENTLENS_AGENT_TRIGGER_CONFIDENCE_MAX",
                settings.agent_expert.trigger_confidence_max,
            ),
            subject_margin_max=env_float(
                "EVENTLENS_AGENT_TRIGGER_SUBJECT_MARGIN_MAX",
                settings.agent_expert.trigger_subject_margin_max,
            ),
        )
        del retriever, router, embedding_client
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        if args.provider == "openai_compatible":
            chat_client = OpenAICompatibleChatClient(
                base_url=args.base_url,
                model=args.model,
                api_key_env=args.api_key_env,
                temperature=env_float(
                    "EVENTLENS_LLM_TEMPERATURE", settings.agent_expert.temperature
                ),
                max_tokens=env_int(
                    "EVENTLENS_LLM_MAX_TOKENS", settings.agent_expert.max_tokens
                ),
                timeout_seconds=env_float("EVENTLENS_LLM_TIMEOUT_SECONDS", 60.0),
                thinking=env_str("EVENTLENS_LLM_THINKING", "disabled"),
                reasoning_effort=env_str(
                    "EVENTLENS_LLM_REASONING_EFFORT", "high"
                ),
            )
        else:
            chat_client = TransformersChatClient(
                model=args.model,
                device=native.device,
                max_tokens=settings.agent_expert.max_tokens,
                cache_folder=native.cache_folder,
                local_files_only=args.local_files_only,
            )
        agent = EventExpertAgent(
            chat_client,
            max_steps=env_int(
                "EVENTLENS_AGENT_MAX_STEPS", settings.agent_expert.max_steps
            ),
            max_content_chars=env_int(
                "EVENTLENS_AGENT_MAX_CONTENT_CHARS",
                settings.agent_expert.max_content_chars,
            ),
        )
        decisions = run_agent_cases(
            agent,
            test_articles,
            predictions,
            routes,
            recalls,
            indices,
        )
        raw_report = evaluate_agent_decisions(test_articles, predictions, indices, decisions)
        if not args.no_verifier:
            verifier = EventChangeVerifier(
                chat_client,
                max_content_chars=env_int(
                    "EVENTLENS_AGENT_MAX_CONTENT_CHARS",
                    settings.agent_expert.max_content_chars,
                ),
            )
            decisions = verify_agent_decisions(
                verifier,
                test_articles,
                predictions,
                recalls,
                indices,
                decisions,
            )
        report = evaluate_agent_decisions(test_articles, predictions, indices, decisions)
        payload = {
            "scope": args.scope,
            "provider": args.provider,
            "model": args.model,
            "selector": {
                "confidence_max": settings.agent_expert.trigger_confidence_max,
                "subject_margin_max": settings.agent_expert.trigger_subject_margin_max,
                "selected_indices": indices,
            },
            "raw_expert_metrics": raw_report.model_dump(),
            "metrics": report.model_dump(),
        }
        if isinstance(chat_client, OpenAICompatibleChatClient):
            payload["api_usage"] = dict(chat_client.usage)
        write_json(args.output, payload)
        write_jsonl(args.trace_output, decisions)
        print(json.dumps(payload, ensure_ascii=False))
        return

    if args.command == "evaluate-event-external":
        train_sets = read_competition_labeled_excel(settings.paths.tagged_train)
        test_sets = read_competition_labeled_excel(settings.paths.tagged_test)
        key = f"{args.scope}_event"
        train_articles = train_sets[key]
        test_articles = test_sets[key]
        manifest, vectors = load_exported_vectors(args.embeddings_dir)
        article_ids = load_exported_article_ids(args.embeddings_dir)
        if manifest.article_count != len(test_articles) or article_ids != [
            row.article_id for row in test_articles
        ]:
            raise ValueError("外部事件评测 embedding 与 tagged test 顺序不一致")

        native = settings.native_embedding
        embedding_client = NativeSentenceTransformerEmbeddingClient(
            model=native.model,
            device=native.device,
            batch_size=native.batch_size,
            normalize_embeddings=native.normalize_embeddings,
            cache_folder=native.cache_folder,
            local_files_only=native.local_files_only,
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
        policy_cfg = getattr(settings.subject_routing, args.scope)
        routes = router.route_from_vectors(
            test_articles,
            vectors,
            scope=args.scope,
            policy=SubjectRoutingPolicy(**policy_cfg.model_dump()),
        )
        subject_codes = [
            [row.accepted_subject_code]
            if row.accepted_subject_code
            else [candidate.subject_code for candidate in row.candidates]
            for row in routes
        ]
        retriever = SubjectConstrainedEventRetriever(
            schema_index,
            embedding_client,
            max_query_chars=settings.event_retrieval.max_query_chars,
        )
        recalls = retriever.recall_from_vectors(
            article_ids,
            vectors,
            subject_codes,
            scope=args.scope,
            top_k=args.top_k,
        )

        event_only_train = [
            row.model_copy(update={"polarity_label": None}) for row in train_articles
        ]
        baseline_model = train_baseline(event_only_train, settings.model.model_dump())
        baseline_predictions = [
            baseline_model.predict_one(row).event_type for row in test_articles
        ]
        embedding_linear_predictions = None
        if args.train_embeddings_dir:
            train_manifest, train_vectors = load_exported_vectors(
                args.train_embeddings_dir
            )
            train_ids = load_exported_article_ids(args.train_embeddings_dir)
            if train_manifest.article_count != len(train_articles) or train_ids != [
                row.article_id for row in train_articles
            ]:
                raise ValueError("训练 embedding 与 tagged train 顺序不一致")
            embedding_linear_predictions = fit_embedding_linear_predictions(
                train_vectors,
                [str(row.event_label) for row in train_articles],
                vectors,
                random_state=settings.model.random_state,
            )
        report = evaluate_external_event_results(
            test_articles,
            baseline_predictions,
            routes,
            recalls,
            scope=args.scope,
            top_k=args.top_k,
            embedding_linear_predictions=embedding_linear_predictions,
            train_articles=train_articles,
            challenge_config=settings.challenge_evaluation.model_dump(),
        )
        write_json(args.output, report.model_dump())
        print(report.model_dump_json())
        return

    if args.command == "evaluate-transformer-event":
        train_sets = read_competition_labeled_excel(settings.paths.tagged_train)
        test_sets = read_competition_labeled_excel(settings.paths.tagged_test)
        key = f"{args.scope}_event"
        train_articles = train_sets[key]
        test_articles = test_sets[key]

        event_only_train = [
            row.model_copy(update={"polarity_label": None}) for row in train_articles
        ]
        baseline_model = train_baseline(event_only_train, settings.model.model_dump())
        baseline_predictions = [
            baseline_model.predict_one(row).event_type for row in test_articles
        ]
        baseline_metrics = transformer_classification_metrics(
            [str(row.event_label) for row in test_articles], baseline_predictions
        )
        transformer_cfg = settings.transformer_event
        gate = (
            transformer_cfg.company_gate_macro_f1
            if args.scope == "company"
            else transformer_cfg.industry_gate_macro_f1
        )
        experiment_config = TransformerTrainingConfig(
            model=args.model,
            max_length=transformer_cfg.max_length,
            max_content_chars=transformer_cfg.max_content_chars,
            batch_size=transformer_cfg.batch_size,
            epochs=transformer_cfg.epochs,
            learning_rate=transformer_cfg.learning_rate,
            weight_decay=transformer_cfg.weight_decay,
            validation_ratio=transformer_cfg.validation_ratio,
            warmup_ratio=transformer_cfg.warmup_ratio,
            label_smoothing=transformer_cfg.label_smoothing,
            class_weight_power=transformer_cfg.class_weight_power,
            early_stopping_patience=transformer_cfg.early_stopping_patience,
            include_subject_fields=transformer_cfg.include_subject_fields,
            gate_macro_f1=gate,
            random_state=settings.model.random_state,
        )
        report = run_transformer_event_experiment(
            train_articles,
            test_articles,
            scope=args.scope,
            config=experiment_config,
            baseline_external_macro_f1=baseline_metrics.macro_f1,
            output_model_dir=args.model_dir,
            local_files_only=args.local_files_only,
            device=settings.native_embedding.device,
        )
        write_json(args.output, report.model_dump())
        print(report.model_dump_json())
        return


def _parse_dependency(value: str) -> DependencyHealth:
    try:
        name, status = value.split(":", 1)
    except ValueError as exc:
        raise ValueError("dependency 格式必须为 name:healthy|required_down|fallback") from exc
    if status == "healthy":
        return DependencyHealth(name=name, healthy=True)
    if status == "required_down":
        return DependencyHealth(name=name, healthy=False, required=True, fallback_available=False)
    if status == "fallback":
        return DependencyHealth(name=name, healthy=False, required=True, fallback_available=True)
    raise ValueError("dependency 状态必须为 healthy|required_down|fallback")


if __name__ == "__main__":
    main()
