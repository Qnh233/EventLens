from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AppInfo(FrozenModel):
    name: str
    version: str
    timezone: str


class PathsConfig(FrozenModel):
    raw_train: str
    raw_test: str
    tagged_train: str
    tagged_test: str
    untagged_train: str
    untagged_test: str
    company_event_schema: str
    industry_event_schema: str
    model_dir: str
    output_dir: str
    data_profile_report: str
    generalization_report: str
    lifecycle_ledger: str
    feedback_store: str
    skill_registry: str
    skill_export_dir: str


class TfidfConfig(FrozenModel):
    analyzer: str
    ngram_range: tuple[int, int]
    min_df: int
    max_features: int


class TextConfig(FrozenModel):
    max_content_chars: int
    tfidf: TfidfConfig


class ClassifierConfig(FrozenModel):
    algorithm: str
    no_event_labels: list[str] = []
    fallback_label: str | None = None
    solver: str = "saga"
    max_iter: int = 500
    tol: float = 0.001
    class_weight: str | None = "balanced"
    alpha: float = 0.0001


class ModelConfig(FrozenModel):
    random_state: int
    text: TextConfig
    event_classifier: ClassifierConfig
    polarity_classifier: ClassifierConfig


class ClusterWeights(FrozenModel):
    text_similarity: float
    entity_match: float
    event_type_match: float
    time_close: float
    key_token_overlap: float


class OllamaEmbeddingConfig(FrozenModel):
    base_url: str
    model: str
    timeout_seconds: float
    batch_size: int
    num_gpu: int | None = None


class NativeEmbeddingConfig(FrozenModel):
    model: str = "BAAI/bge-m3"
    device: str = "cuda"
    batch_size: int = 16
    normalize_embeddings: bool = True
    cache_folder: str | None = None
    local_files_only: bool = False


class EmbeddingExportConfig(FrozenModel):
    max_content_chars: int = 1600
    chunk_size: int = 1024
    output_root: str = "artifacts/embeddings"


class SubjectScopeRoutingConfig(FrozenModel):
    top_k: int = 3
    exact_alias_hard_route: bool = False
    bge_hard_route: bool = False
    score_threshold: float = 1.0
    margin_threshold: float = 1.0


class SubjectRoutingConfig(FrozenModel):
    max_query_chars: int = 1600
    min_alias_chars: int = 2
    company: SubjectScopeRoutingConfig
    industry: SubjectScopeRoutingConfig


class HardExampleConfig(FrozenModel):
    subject_margin_threshold: float = 0.05
    event_margin_threshold: float = 0.05
    max_examples: int = 5000


class CandidateEdgeEvaluationConfig(FrozenModel):
    minimum_eligible_recall: float = 0.95


class ClusterSemanticConfig(FrozenModel):
    enabled: bool = False
    fail_open: bool = True
    candidate_threshold: float = 0.45
    similarity_threshold: float = 0.92093
    max_text_chars: int = 1200
    cache_path: str = "artifacts/cache/bge_embeddings.sqlite3"
    embedding: OllamaEmbeddingConfig


class ClusterConfig(FrozenModel):
    threshold: float
    weights: ClusterWeights
    time_window_days: int
    top_k: int
    semantic: ClusterSemanticConfig


class SourceAuthorityConfig(FrozenModel):
    official_keywords: list[str]
    company_keywords: list[str]
    mainstream_keywords: list[str]
    general_keywords: list[str]
    self_media_keywords: list[str]


class CredibilityScores(FrozenModel):
    official: float
    company: float
    mainstream: float
    general: float
    self_media: float
    unknown: float
    low_quality: float


class CredibilityWeights(FrozenModel):
    source_authority: float
    multi_source_consistency: float
    official_endorsement: float
    content_completeness: float
    recency_score: float


class AlertThresholds(FrozenModel):
    credibility_high: float
    credibility_medium: float
    severity_high: float
    severity_medium: float


class CredibilityConfig(FrozenModel):
    source_authority: SourceAuthorityConfig
    scores: CredibilityScores
    credibility_weights: CredibilityWeights
    severity_baseline: dict[str, float]
    alert_thresholds: AlertThresholds


class EvaluationConfig(FrozenModel):
    time_train_ratio: float
    group_holdout_ratio: float
    random_state: int


class DuplicatePairConfig(FrozenModel):
    negative_ratio: float
    max_positive_pairs_per_group: int
    time_window_days: int
    max_text_chars: int
    seed: int
    subject_lead_chars: int = 240
    min_subject_alias_chars: int = 2
    require_resolved_subject_for_negatives: bool = True


class DuplicatePairEvaluationConfig(FrozenModel):
    calibration_ratio: float
    seed: int


class ClusterBenchmarkConfig(FrozenModel):
    max_articles: int
    candidate_thresholds: list[float]
    semantic_thresholds: list[float]
    top_ks: list[int]
    minimum_b_cubed_f1_gain: float
    minimum_pairwise_recall_gain: float
    stress_sample_size: int


class EventRetrievalConfig(FrozenModel):
    top_k: int
    max_query_chars: int
    sample_limit: int
    embedding: OllamaEmbeddingConfig


class LifecycleConfig(FrozenModel):
    official_authority_threshold: float
    company_authority_threshold: float
    mainstream_authority_threshold: float
    corroborated_source_count: int
    base_evidence_gain: float
    independent_source_bonus: float
    official_confirmation_bonus: float
    clarification_confidence_bonus: float
    dispute_confidence_factor: float
    support_keywords: list[str]
    refutation_keywords: list[str]
    escalation_keywords: list[str]
    resolution_keywords: list[str]


class LearningConfig(FrozenModel):
    low_confidence_threshold: float
    high_severity_threshold: float
    min_feedback_count: int
    min_macro_f1_gain: float
    max_critical_error_regression: float
    require_human_approval: bool


class EvidenceControlConfig(FrozenModel):
    high_risk_levels: list[str]
    minimum_high_risk_independent_sources: int = 2
    official_bypass_authority: float = 0.95


class RuntimeControlConfig(FrozenModel):
    min_workers: int = 1
    max_workers: int = 4
    scale_up_queue_depth: int = 500
    scale_down_queue_depth: int = 50
    degraded_after_failures: int = 3
    stop_after_failures: int = 5
    retry_backoff_seconds: list[int] = [30, 120, 300]


class ChallengeEvaluationConfig(FrozenModel):
    rare_event_max_train_count: int = 20
    long_tail_source_max_train_count: int = 5
    long_text_percentile: float = 0.75


class AgentExpertConfig(FrozenModel):
    enabled: bool = False
    provider: str = "local_transformers"
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    base_url: str = ""
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 256
    max_steps: int = 4
    max_content_chars: int = 4000
    trigger_confidence_max: float = 0.65
    trigger_subject_margin_max: float = 0.05


class TransformerEventConfig(FrozenModel):
    model: str = "hfl/chinese-macbert-base"
    max_length: int = 384
    max_content_chars: int = 1800
    batch_size: int = 16
    epochs: int = 6
    learning_rate: float = 0.00002
    weight_decay: float = 0.01
    validation_ratio: float = 0.2
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.05
    class_weight_power: float = 0.5
    early_stopping_patience: int = 2
    include_subject_fields: bool = False
    company_gate_macro_f1: float = 0.78
    industry_gate_macro_f1: float = 0.85


class Settings(FrozenModel):
    app: AppInfo
    paths: PathsConfig
    model: ModelConfig
    cluster: ClusterConfig
    credibility: CredibilityConfig
    evaluation: EvaluationConfig
    duplicate_pairs: DuplicatePairConfig
    duplicate_pair_evaluation: DuplicatePairEvaluationConfig
    cluster_benchmark: ClusterBenchmarkConfig
    event_retrieval: EventRetrievalConfig
    native_embedding: NativeEmbeddingConfig
    embedding_export: EmbeddingExportConfig
    subject_routing: SubjectRoutingConfig
    hard_examples: HardExampleConfig
    candidate_edge_evaluation: CandidateEdgeEvaluationConfig
    lifecycle: LifecycleConfig
    learning: LearningConfig
    evidence_control: EvidenceControlConfig
    runtime_control: RuntimeControlConfig
    challenge_evaluation: ChallengeEvaluationConfig
    agent_expert: AgentExpertConfig
    transformer_event: TransformerEventConfig


@lru_cache(maxsize=8)
def load_settings(path: str | Path = "configs/app.yaml") -> Settings:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return Settings.model_validate(payload)
