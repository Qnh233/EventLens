from datetime import datetime

from eventlens.evaluation import evaluate_split, group_holdout_split, render_generalization_report, time_split
from eventlens.schema import ArticleRecord


def test_generalization_splits_are_non_empty():
    articles = [
        ArticleRecord(article_id="A1", publish_time=datetime(2026, 1, 1), entity="公司A", source="来源1"),
        ArticleRecord(article_id="A2", publish_time=datetime(2026, 1, 2), entity="公司B", source="来源2"),
        ArticleRecord(article_id="A3", publish_time=datetime(2026, 1, 3), entity="公司C", source="来源3"),
        ArticleRecord(article_id="A4", publish_time=datetime(2026, 1, 4), entity="公司D", source="来源4"),
    ]

    train, valid = time_split(articles)
    assert train and valid

    train, valid = group_holdout_split(articles, "entity")
    assert train and valid
    assert "泛化评测报告" in render_generalization_report(articles)


def test_evaluate_split_returns_real_metrics():
    train = [
        ArticleRecord(article_id="T1", title="收到监管处罚", content="公司违规被处罚", event_label="监管处罚"),
        ArticleRecord(article_id="T2", title="发生重大诉讼", content="公司被起诉", event_label="重大诉讼"),
        ArticleRecord(article_id="T3", title="监管立案调查", content="监管机构立案", event_label="监管处罚"),
        ArticleRecord(article_id="T4", title="法院判决", content="诉讼案件判决", event_label="重大诉讼"),
    ]
    valid = [
        ArticleRecord(article_id="V1", title="监管处罚决定", content="公司收到处罚", event_label="监管处罚"),
        ArticleRecord(article_id="V2", title="诉讼仲裁进展", content="重大诉讼进展", event_label="重大诉讼"),
    ]

    result = evaluate_split(train, valid)

    assert result["status"] == "ok"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert len(result["confusion_matrix"]) == len(result["labels"])

