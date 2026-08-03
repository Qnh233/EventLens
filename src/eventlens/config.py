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
    model_dir: str
    output_dir: str
    data_profile_report: str
    generalization_report: str


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


class ClusterConfig(FrozenModel):
    threshold: float
    weights: ClusterWeights
    time_window_days: int
    top_k: int


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


class Settings(FrozenModel):
    app: AppInfo
    paths: PathsConfig
    model: ModelConfig
    cluster: ClusterConfig
    credibility: CredibilityConfig
    evaluation: EvaluationConfig


@lru_cache(maxsize=8)
def load_settings(path: str | Path = "configs/app.yaml") -> Settings:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    return Settings.model_validate(payload)
