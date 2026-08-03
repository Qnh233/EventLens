import pandas as pd

from eventlens.io import profile_articles, read_articles_excel


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

