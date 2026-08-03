from eventlens.preprocess import clean_text, normalize_entity, normalize_polarity, parse_datetime


def test_clean_text_collapses_whitespace():
    assert clean_text("  A\n  B\tC  ") == "A B C"


def test_parse_datetime_handles_invalid_value():
    assert parse_datetime("bad-date") is None
    assert parse_datetime("2026-04-12").year == 2026


def test_normalize_entity_removes_common_suffix():
    assert normalize_entity("示例科技股份有限公司") == "示例科技"


def test_normalize_polarity_uses_context_keywords():
    assert normalize_polarity("", "公司收到监管处罚") == "负面"
    assert normalize_polarity("", "公司取得技术突破") == "正面"

