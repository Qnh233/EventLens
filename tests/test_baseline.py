from datetime import datetime

from eventlens.baseline import predict_articles, train_baseline
from eventlens.schema import ArticleRecord


def test_train_baseline_predicts_with_labels():
    articles = [
        ArticleRecord(
            article_id="A1",
            title="示例科技收到监管处罚",
            publish_time=datetime(2026, 4, 12),
            source="深交所公告",
            content="公司因为信息披露违规收到处罚。",
            entity="示例科技",
            event_label="监管处罚",
            polarity_label="负面",
        ),
        ArticleRecord(
            article_id="A2",
            title="样本公司发生重大诉讼",
            publish_time=datetime(2026, 4, 13),
            source="证券时报",
            content="样本公司涉及重大诉讼。",
            entity="样本公司",
            event_label="重大诉讼",
            polarity_label="负面",
        ),
        ArticleRecord(
            article_id="A3",
            title="创新药获得研发突破",
            publish_time=datetime(2026, 4, 14),
            source="公司公告",
            content="公司研发取得技术突破。",
            entity="创新药业",
            event_label="技术突破",
            polarity_label="正面",
        ),
    ]

    model = train_baseline(articles)
    predictions = predict_articles(articles, model=model)

    assert len(predictions) == 3
    assert all(pred.has_event for pred in predictions)
    assert all(0.0 <= pred.classifier_confidence <= 1.0 for pred in predictions)
    assert all(0.0 <= pred.polarity_confidence <= 1.0 for pred in predictions)

