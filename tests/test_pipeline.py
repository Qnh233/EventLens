from datetime import datetime

from eventlens.pipeline import run_pipeline, write_pipeline_outputs
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
    assert len(result["lifecycles"]) == 1
    assert result["lifecycles"][0].snapshots
    assert "learning_signals" in result
    assert "cluster_decisions" in result


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


def test_pipeline_outputs_include_auditable_lifecycle_ledger(tmp_path):
    articles = [
        ArticleRecord(
            article_id="A1",
            title="公司收到监管问询函",
            publish_time=datetime(2026, 4, 12),
            source="深交所",
            content="深交所正式向公司发出问询函。",
            entity="示例科技",
        )
    ]
    result = run_pipeline(articles)
    output_dir = tmp_path / "run"
    ledger_path = tmp_path / "ledger" / "event_lifecycle.jsonl"

    write_pipeline_outputs(output_dir, result, lifecycle_ledger_path=ledger_path)

    assert (output_dir / "event_lifecycle.jsonl").exists()
    assert (output_dir / "learning_signals.jsonl").exists()
    assert (output_dir / "cluster_decision.jsonl").exists()
    assert ledger_path.exists()

    write_pipeline_outputs(output_dir, result, lifecycle_ledger_path=ledger_path)

    from eventlens.lifecycle import EventLifecycleLedger

    ledger = EventLifecycleLedger(ledger_path)
    assert len(ledger.read_all()) == 2
    assert ledger.latest()[result["lifecycles"][0].event_cluster_id].version == 2

