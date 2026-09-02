from eventlens.config import load_settings


def test_load_settings_reads_single_central_config():
    settings = load_settings("configs/app.yaml")

    assert settings.app.name == "EventLens"
    assert settings.app.version == "0.7.0"
    assert settings.model.text.max_content_chars == 1200
    assert settings.model.text.tfidf.max_features == 30000
    assert settings.cluster.threshold == 0.72
    assert settings.cluster.semantic.enabled is False
    assert settings.cluster.semantic.similarity_threshold == 0.92093
    assert settings.cluster.semantic.embedding.num_gpu == 0
    assert settings.credibility.alert_thresholds.credibility_high == 0.75
    assert settings.evaluation.time_train_ratio == 0.8
    assert settings.paths.model_dir == "artifacts/models"
    assert settings.paths.tagged_train.endswith("news_with_tags_train.xlsx")
    assert settings.paths.untagged_test.endswith("news_without_tags_test.xlsx")
    assert settings.paths.company_event_schema.endswith("事件类型_标的.json")
    assert settings.lifecycle.corroborated_source_count == 2
    assert settings.learning.min_feedback_count == 3
    assert settings.learning.require_human_approval is True
    assert settings.paths.lifecycle_ledger.endswith("event_lifecycle.jsonl")
    assert settings.duplicate_pair_evaluation.calibration_ratio == 0.5
    assert settings.duplicate_pairs.negative_ratio == 1.0
    assert settings.duplicate_pairs.require_resolved_subject_for_negatives is True
    assert settings.event_retrieval.embedding.model == "bge-m3:latest"
    assert settings.event_retrieval.embedding.batch_size == 8
    assert settings.cluster_benchmark.candidate_thresholds == [0.4, 0.45, 0.5]
    assert settings.cluster_benchmark.stress_sample_size == 10000
    assert settings.native_embedding.model == "BAAI/bge-m3"
    assert settings.native_embedding.batch_size == 16
    assert settings.embedding_export.chunk_size == 1024
    assert settings.subject_routing.company.bge_hard_route is True
    assert settings.subject_routing.company.score_threshold == 0.426251
    assert settings.subject_routing.industry.bge_hard_route is False
    assert settings.evidence_control.minimum_high_risk_independent_sources == 2
    assert settings.runtime_control.max_workers == 4
    assert settings.challenge_evaluation.rare_event_max_train_count == 20
    assert settings.transformer_event.model == "hfl/chinese-macbert-base"
    assert settings.transformer_event.company_gate_macro_f1 == 0.78
    assert settings.transformer_event.include_subject_fields is False
