from __future__ import annotations

from datetime import datetime
from pathlib import Path


from eventlens.config import load_settings
from eventlens.schema import AlertOutput, ArticleRecord, CredibilityBreakdown, EventCluster


def load_credibility_config(path: str | Path = "configs/app.yaml") -> dict:
    return load_settings(path).credibility.model_dump()


def source_authority_score(source: str, cfg: dict) -> float:
    source = source or ""
    keywords = cfg["source_authority"]
    scores = cfg["scores"]
    if any(word in source for word in keywords["official_keywords"]):
        return float(scores["official"])
    if any(word in source for word in keywords["company_keywords"]):
        return float(scores["company"])
    if any(word in source for word in keywords["mainstream_keywords"]):
        return float(scores["mainstream"])
    if any(word in source for word in keywords["general_keywords"]):
        return float(scores["general"])
    if any(word in source for word in keywords["self_media_keywords"]):
        return float(scores["self_media"])
    return float(scores["unknown"])


def build_alerts(
    articles: list[ArticleRecord],
    clusters: list[EventCluster],
    cfg: dict | None = None,
    now: datetime | None = None,
) -> list[AlertOutput]:
    config = cfg or load_credibility_config()
    article_map = {article.article_id: article for article in articles}
    alerts: list[AlertOutput] = []
    for idx, cluster in enumerate(clusters, start=1):
        cluster_articles = [article_map[article_id] for article_id in cluster.article_ids if article_id in article_map]
        breakdown = credibility_breakdown(cluster_articles, config, now)
        credibility = credibility_score(breakdown, config)
        severity = severity_score(cluster.event_type, config)
        direction = _cluster_direction(cluster_articles)
        risk_level = alert_level_from_matrix(credibility, severity, direction, config)
        evidence_sources = [
            {
                "article_id": article.article_id,
                "source": article.source or "未知来源",
                "date": article.publish_time.date().isoformat() if article.publish_time else "",
                "type": "官方/权威" if source_authority_score(article.source, config) >= 0.9 else "媒体/其他",
            }
            for article in cluster_articles[:5]
        ]
        alerts.append(
            AlertOutput(
                alert_id=f"ALT-{idx:05d}",
                risk_level=risk_level,
                impact_direction=direction,
                event_cluster_id=cluster.event_cluster_id,
                company=cluster.main_company,
                event_type=cluster.event_type,
                event_summary=_summary(cluster),
                credibility_score=round(credibility, 4),
                severity_score=round(severity, 4),
                credibility_breakdown=breakdown,
                evidence_sources=evidence_sources,
                push_reason=_push_reason(credibility, severity, direction, risk_level),
                related_article_count=len(cluster.article_ids),
            )
        )
    return alerts


def credibility_breakdown(
    articles: list[ArticleRecord],
    cfg: dict,
    now: datetime | None = None,
) -> CredibilityBreakdown:
    if not articles:
        return CredibilityBreakdown(
            source_authority=0.0,
            multi_source_consistency=0.0,
            official_endorsement=0.0,
            content_completeness=0.0,
            recency_score=0.0,
        )
    source_scores = [source_authority_score(article.source, cfg) for article in articles]
    distinct_sources = {article.source for article in articles if article.source}
    official = 1.0 if max(source_scores) >= 0.9 else 0.0
    completeness = sum(_content_complete(article) for article in articles) / len(articles)
    return CredibilityBreakdown(
        source_authority=sum(source_scores) / len(source_scores),
        multi_source_consistency=min(1.0, len(distinct_sources) / 3),
        official_endorsement=official,
        content_completeness=completeness,
        recency_score=_recency_score(articles, now),
        duplicate_penalty=_duplicate_penalty(articles),
        conflict_penalty=0.0,
    )


def credibility_score(breakdown: CredibilityBreakdown, cfg: dict) -> float:
    weights = cfg["credibility_weights"]
    score = (
        weights["source_authority"] * breakdown.source_authority
        + weights["multi_source_consistency"] * breakdown.multi_source_consistency
        + weights["official_endorsement"] * breakdown.official_endorsement
        + weights["content_completeness"] * breakdown.content_completeness
        + weights["recency_score"] * breakdown.recency_score
        - breakdown.duplicate_penalty
        - breakdown.conflict_penalty
    )
    return min(1.0, max(0.0, score))


def severity_score(event_type: str, cfg: dict) -> float:
    baseline = cfg["severity_baseline"]
    for key, score in baseline.items():
        if key in event_type:
            return float(score)
    return float(baseline.get("其他事件", 0.35))


def alert_level_from_matrix(credibility: float, severity: float, direction: str, cfg: dict) -> str:
    thresholds = cfg["alert_thresholds"]
    cred_high = credibility >= thresholds["credibility_high"]
    cred_mid = credibility >= thresholds["credibility_medium"]
    sev_high = severity >= thresholds["severity_high"]
    sev_mid = severity >= thresholds["severity_medium"]
    if direction == "正面":
        if cred_high and sev_high:
            return "高价值机会"
        if cred_mid and sev_mid:
            return "中价值机会"
        return "机会观察"
    if direction == "中性":
        if cred_high and sev_high:
            return "高关注"
        if cred_mid and sev_mid:
            return "中关注"
        return "常规披露"
    if cred_high and sev_high:
        return "高风险"
    if not cred_mid and sev_high:
        return "待核实重点关注"
    if cred_high and not sev_mid:
        return "常规披露"
    if cred_mid and sev_mid:
        return "中风险"
    return "观察"


def _content_complete(article: ArticleRecord) -> float:
    fields = [article.title, article.publish_time, article.source, article.content, article.entity]
    return sum(1 for field in fields if field) / len(fields)


def _recency_score(articles: list[ArticleRecord], now: datetime | None) -> float:
    dated = [article.publish_time for article in articles if article.publish_time]
    if not dated:
        return 0.5
    anchor = now or max(dated)
    min_days = min(abs((anchor - dt).total_seconds()) / 86400 for dt in dated)
    return max(0.0, 1.0 - min(min_days, 30) / 30)


def _duplicate_penalty(articles: list[ArticleRecord]) -> float:
    if not articles:
        return 0.0
    duplicate_count = sum(1 for article in articles if str(article.duplicate_flag).strip() in {"1", "true", "True", "是"})
    return min(0.2, duplicate_count / len(articles) * 0.1)


def _summary(cluster: EventCluster) -> str:
    company = cluster.main_company or "相关公司"
    return f"{company}发生{cluster.event_type}事件，当前聚合到{len(cluster.article_ids)}篇相关报道。"


def _cluster_direction(articles: list[ArticleRecord]) -> str:
    labels = [article.polarity_label for article in articles if article.polarity_label]
    if not labels:
        text = " ".join(f"{article.title} {article.content}" for article in articles)
        from eventlens.preprocess import normalize_polarity

        return normalize_polarity("", text)
    counts = {label: labels.count(label) for label in {"正面", "负面", "中性"}}
    return max(counts, key=counts.get)


def _push_reason(credibility: float, severity: float, direction: str, risk_level: str) -> str:
    return f"可信度={credibility:.2f}，影响强度={severity:.2f}，方向={direction}，矩阵判定为{risk_level}。"

