from datetime import datetime

from eventlens.pipeline import run_pipeline
from eventlens.schema import ArticleRecord


def test_pipeline_outputs_alert_for_event():
    articles = [
        ArticleRecord(
            article_id="A1",
            title="示例科技收到监管问询函",
            publish_time=datetime(2026, 4, 12),
            source="深交所公告",
            content="示例科技收到深交所问询函，要求说明信息披露事项。",
            entity="示例科技",
        ),
        ArticleRecord(
            article_id="A2",
            title="示例科技回应监管问询",
            publish_time=datetime(2026, 4, 13),
            source="证券时报",
            content="示例科技回应监管问询，事件继续发酵。",
            entity="示例科技",
        ),
    ]

    result = run_pipeline(articles)

    assert result["predictions"][0].has_event is True
    assert len(result["clusters"]) == 1
    assert result["alerts"][0].risk_level in {"高风险", "中风险"}


def test_positive_event_uses_opportunity_semantics():
    articles = [
        ArticleRecord(
            article_id="P1",
            title="创新药获得批准并实现技术突破",
            publish_time=datetime(2026, 4, 12),
            source="公司公告",
            content="创新药获得批准，公司研发取得技术突破。",
            entity="创新药业",
        )
    ]

    alert = run_pipeline(articles)["alerts"][0]

    assert alert.impact_direction == "正面"
    assert "机会" in alert.risk_level
    assert "风险" not in alert.risk_level

