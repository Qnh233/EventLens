import pandas as pd

from eventlens.io import profile_articles, read_articles_excel, read_competition_labeled_excel


def test_read_articles_excel_maps_chinese_columns(tmp_path):
    path = tmp_path / "sample.xlsx"
    pd.DataFrame(
        [
            {
                "文章id": "A1",
                "文章标题": "示例科技收到问询函",
                "发布日期": "2026-04-12",
                "来源网站": "深交所公告",
                "正文文本": "示例科技收到深交所问询函。",
                "实体": "示例科技股份有限公司",
                "事件": "监管处罚",
                "事件情感正负面": "负面",
            }
        ]
    ).to_excel(path, index=False)

    articles = read_articles_excel(path)

    assert articles[0].article_id == "A1"
    assert articles[0].entity == "示例科技"
    assert articles[0].event_label == "监管处罚"
    assert profile_articles(articles)["article_count"] == 1


def test_read_competition_workbook_maps_real_headers_and_separates_tasks(tmp_path):
    path = tmp_path / "competition.xlsx"
    company = pd.DataFrame(
        [
            {
                "article_file_id": "C1",
                "article_title": "公司发布新产品",
                "article_publish_time": "2026-04-12",
                "article_source": "公司公告",
                "content": "公司发布新产品。",
                "trading_code": "000001",
                "secu_abbr": "示例股份",
                "industry_code": "I01",
                "industry_name": "示例行业",
                "event_name": "产品技术创新",
                "event_emotion": "正面",
                "event_impact_analysis": "提升竞争力",
                "duplication_id": "D-1",
            }
        ]
    )
    duplicate = pd.DataFrame(
        [
            {
                "article_file_id": "D1",
                "article_title": "公司新品发布",
                "article_source": "媒体",
                "content": "公司发布新产品。",
                "article_publish_time": "2026-04-12",
                "duplication_id": "D-1",
            }
        ]
    )
    with pd.ExcelWriter(path) as writer:
        company.to_excel(writer, sheet_name="个股新闻", index=False)
        duplicate.to_excel(writer, sheet_name="个股重复新闻", index=False)

    datasets = read_competition_labeled_excel(path)

    event = datasets["company_event"][0]
    repeat = datasets["company_duplicate"][0]
    assert event.article_id == "C1"
    assert event.trading_code == "000001"
    assert event.entity == "示例股份"
    assert event.industry_code == "I01"
    assert event.event_label == "产品技术创新"
    assert event.duplication_id == "D-1"
    assert event.sheet_name == "个股新闻"
    assert repeat.task_scope == "company_duplicate"
    assert repeat.event_label is None
    assert repeat.duplication_id == "D-1"
    assert profile_articles(datasets["company_event"])["missing_subject"] == 0


def test_read_articles_excel_supports_nrows(tmp_path):
    path = tmp_path / "sample_rows.xlsx"
    pd.DataFrame(
        {
            "article_file_id": ["A1", "A2", "A3"],
            "article_title": ["一", "二", "三"],
        }
    ).to_excel(path, index=False)

    articles = read_articles_excel(path, nrows=2)

    assert [article.article_id for article in articles] == ["A1", "A2"]

