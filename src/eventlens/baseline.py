from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from eventlens.config import load_settings
from eventlens.preprocess import build_model_text, evidence_sentence, normalize_polarity
from eventlens.schema import ArticleRecord, EventPrediction

EVENT_KEYWORDS = {
    "监管处罚": ("处罚", "违规", "监管", "立案", "问询函", "关注函"),
    "重大诉讼": ("诉讼", "仲裁", "起诉", "判决"),
    "财务造假": ("财务造假", "虚增", "虚假记载", "会计差错"),
    "退市风险": ("退市", "ST", "暂停上市"),
    "业绩预告": ("业绩", "净利润", "营收", "亏损", "增长"),
    "股权质押": ("质押", "解除质押", "冻结"),
    "高管变动": ("董事长", "总经理", "辞职", "任命", "高管"),
    "技术突破": ("专利", "研发", "突破", "获批", "临床"),
    "供应链变化": ("供应", "订单", "中标", "采购", "客户"),
    "政策影响": ("政策", "补贴", "关税", "限产", "监管办法"),
}


@dataclass
class TrainedBaseline:
    event_model: Pipeline | DummyClassifier
    polarity_model: Pipeline | DummyClassifier | None
    config: dict

    def save(self, model_dir: str | Path) -> None:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path / "baseline.joblib")

    @staticmethod
    def load(model_dir: str | Path) -> "TrainedBaseline":
        return joblib.load(Path(model_dir) / "baseline.joblib")

    def predict_one(self, article: ArticleRecord) -> EventPrediction:
        max_chars = self.config["text"]["max_content_chars"]
        text = build_model_text(article, max_content_chars=max_chars)
        event_type, event_conf = _predict_label(self.event_model, [text])
        no_event_labels = set(self.config["event_classifier"].get("no_event_labels", []))

        polarity = "中性"
        polarity_conf = 0.0
        if self.polarity_model is not None:
            polarity, polarity_conf = _predict_label(self.polarity_model, [text])
        if polarity not in {"正面", "负面", "中性"}:
            polarity = normalize_polarity(polarity, text)

        has_event = bool(event_type and event_type not in no_event_labels)
        evidence = evidence_sentence(article, event_type)
        return EventPrediction(
            article_id=article.article_id,
            has_event=has_event,
            event_type=event_type or "无事件",
            event_polarity=polarity,
            classifier_confidence=event_conf,
            polarity_confidence=polarity_conf,
            event_subject=article.entity or None,
            event_time=article.publish_time,
            impact_target=article.entity or article.industry or None,
            impact_direction=polarity,
            impact_description=article.impact_analysis or "",
            evidence_sentence=evidence,
            extraction_confidence=0.5 if evidence else 0.0,
        )


def load_config(path: str | Path = "configs/app.yaml") -> dict:
    return load_settings(path).model.model_dump()


def train_baseline(articles: list[ArticleRecord], config: dict | None = None) -> TrainedBaseline:
    cfg = config or load_config()
    event_samples = [(a, a.event_label) for a in articles if a.event_label]
    if not event_samples:
        raise ValueError("缺少 event_label，无法训练事件分类 baseline")

    max_chars = cfg["text"]["max_content_chars"]
    event_texts = [build_model_text(a, max_chars) for a, _ in event_samples]
    event_labels = [label for _, label in event_samples]
    event_algorithm = cfg["event_classifier"].get("algorithm", "logistic_regression")
    event_model = _fit_text_classifier(event_texts, event_labels, cfg, event_algorithm)

    polarity_samples = [(a, a.polarity_label) for a in articles if a.polarity_label]
    polarity_model = None
    if polarity_samples:
        polarity_texts = [build_model_text(a, max_chars) for a, _ in polarity_samples]
        polarity_labels = [normalize_polarity(label) for _, label in polarity_samples]
        polarity_algorithm = cfg["polarity_classifier"].get("algorithm", "logistic_regression")
        polarity_model = _fit_text_classifier(polarity_texts, polarity_labels, cfg, polarity_algorithm)

    return TrainedBaseline(event_model=event_model, polarity_model=polarity_model, config=cfg)


def heuristic_predict(article: ArticleRecord, config: dict | None = None) -> EventPrediction:
    cfg = config or load_config()
    text = build_model_text(article, cfg["text"]["max_content_chars"])
    event_type = "无事件"
    confidence = 0.35
    for candidate, keywords in EVENT_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            event_type = candidate
            confidence = 0.62
            break
    polarity = normalize_polarity(article.polarity_label, text)
    evidence = evidence_sentence(article, event_type)
    return EventPrediction(
        article_id=article.article_id,
        has_event=event_type != "无事件",
        event_type=event_type,
        event_polarity=polarity,
        classifier_confidence=confidence,
        polarity_confidence=0.45,
        event_subject=article.entity or None,
        event_time=article.publish_time,
        impact_target=article.entity or article.industry or None,
        impact_direction=polarity,
        impact_description=article.impact_analysis or "",
        evidence_sentence=evidence,
        extraction_confidence=0.45 if evidence else 0.0,
    )


def predict_articles(
    articles: Iterable[ArticleRecord],
    model: TrainedBaseline | None = None,
    config: dict | None = None,
) -> list[EventPrediction]:
    return [
        model.predict_one(article) if model else heuristic_predict(article, config)
        for article in articles
    ]


def _fit_text_classifier(
    texts: list[str],
    labels: list[str],
    cfg: dict,
    algorithm: str,
) -> Pipeline | DummyClassifier:
    if len(set(labels)) == 1:
        dummy = DummyClassifier(strategy="constant", constant=labels[0])
        dummy.fit([[0]] * len(labels), labels)
        return dummy

    tfidf_cfg = dict(cfg["text"]["tfidf"])
    tfidf_cfg["ngram_range"] = tuple(tfidf_cfg["ngram_range"])
    if algorithm == "lightgbm":
        from lightgbm import LGBMClassifier

        classifier = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.08,
            num_leaves=31,
            class_weight="balanced",
            random_state=cfg.get("random_state", 42),
            verbose=-1,
        )
    else:
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=cfg.get("random_state", 42),
        )
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**tfidf_cfg)),
            ("clf", classifier),
        ]
    )
    pipe.fit(texts, labels)
    return pipe


def _predict_label(model: Pipeline | DummyClassifier, texts: list[str]) -> tuple[str, float]:
    if isinstance(model, DummyClassifier):
        label = str(model.predict([[0]])[0])
        return label, 1.0

    label = str(model.predict(texts)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)[0]
        return label, float(max(proba))
    return label, 0.0
