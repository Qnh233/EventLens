from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import statistics
import tempfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

COLUMN_ALIASES: dict[str, set[str]] = {
    "article_id": {
        "article_id",
        "article_file_id",
        "id",
        "文章id",
        "文章ID",
        "资讯id",
        "资讯ID",
    },
    "title": {"title", "article_title", "文章标题", "标题"},
    "publish_time": {
        "publish_time",
        "article_publish_time",
        "publish_date",
        "date",
        "发布日期",
        "发布时间",
        "时间",
    },
    "source": {"source", "article_source", "source_site", "来源网站", "来源", "媒体"},
    "content": {"content", "正文文本", "正文", "文章内容", "文本"},
    "entity": {"entity", "secu_abbr", "实体", "公司", "关联企业", "事件关联企业"},
    "trading_code": {"trading_code", "证券代码", "股票代码"},
    "industry": {"industry", "industry_name", "行业", "板块"},
    "industry_code": {"industry_code", "行业代码"},
    "event_label": {
        "event",
        "event_label",
        "event_name",
        "事件",
        "事件类型",
        "事件类型名称",
    },
    "polarity_label": {
        "polarity",
        "event_polarity",
        "polarity_label",
        "event_emotion",
        "事件情感正负面",
        "情感",
        "影响方向",
    },
    "impact_analysis": {
        "impact_analysis",
        "event_impact_analysis",
        "事件影响分析",
        "影响分析",
    },
    "duplicate_flag": {"duplicate_flag", "重复性标志", "是否重复", "重复标志"},
    "duplication_id": {"duplication_id", "重复组ID", "重复组id"},
}

PROVENANCE_HEADER_KEYWORDS = (
    "公开",
    "内部",
    "脱敏",
    "数据来源",
    "来源类型",
    "数据类型",
    "provenance",
    "dataset_type",
)

SOURCE_CATEGORY_KEYWORDS = {
    "官方监管": ("证监会", "上交所", "深交所", "北交所", "交易所", "监管"),
    "公司披露": ("公司公告", "官方网站", "官网", "年报", "半年报", "季报", "公告"),
    "主流财经媒体": ("证券时报", "上海证券报", "中国证券报", "证券日报", "财新", "第一财经"),
    "综合权威媒体": ("新华社", "人民日报", "央视", "经济日报"),
    "社区或自媒体": ("雪球", "股吧", "自媒体", "公众号"),
}


def _xlsx_structure(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        entries = {entry.filename: entry for entry in archive.infolist()}
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
        }
        sheets: list[dict] = []
        for node in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
            relation_id = node.attrib[f"{{{REL_NS}}}id"]
            target = rel_map[relation_id].replace("\\", "/")
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            sheet_entry = entries[sheet_path]
            dimension = ""
            with archive.open(sheet_path) as stream:
                for _, element in ET.iterparse(stream, events=("start",)):
                    if element.tag == f"{{{MAIN_NS}}}dimension":
                        dimension = element.attrib.get("ref", "")
                        break
            sheets.append(
                {
                    "name": node.attrib.get("name", ""),
                    "path": sheet_path,
                    "dimension": dimension,
                    "compressed_bytes": sheet_entry.compress_size,
                    "raw_bytes": sheet_entry.file_size,
                }
            )
        shared = entries.get("xl/sharedStrings.xml")
        return {
            "file": path.name,
            "file_bytes": path.stat().st_size,
            "shared_strings_compressed_bytes": shared.compress_size if shared else 0,
            "shared_strings_raw_bytes": shared.file_size if shared else 0,
            "sheets": sheets,
        }


class SharedStringStore:
    """将大型 sharedStrings 流式落到临时 SQLite，避免占用数百 MB 内存。"""

    def __init__(self, archive: zipfile.ZipFile):
        self.archive = archive
        self.temp_dir = tempfile.TemporaryDirectory(prefix="eventlens-xlsx-")
        self.db_path = Path(self.temp_dir.name) / "shared_strings.sqlite"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA locking_mode=EXCLUSIVE")
        self.connection.execute("CREATE TABLE strings (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        self._build()

    def _build(self) -> None:
        if "xl/sharedStrings.xml" not in self.archive.namelist():
            return
        batch: list[tuple[int, str]] = []
        index = 0
        with self.archive.open("xl/sharedStrings.xml") as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != f"{{{MAIN_NS}}}si":
                    continue
                text = "".join(
                    node.text or "" for node in element.iter(f"{{{MAIN_NS}}}t")
                )
                batch.append((index, text))
                index += 1
                element.clear()
                if len(batch) >= 5000:
                    self.connection.executemany("INSERT INTO strings VALUES (?, ?)", batch)
                    batch.clear()
        if batch:
            self.connection.executemany("INSERT INTO strings VALUES (?, ?)", batch)
        self.connection.commit()

    def fetch_many(self, indexes: set[int]) -> dict[int, str]:
        if not indexes:
            return {}
        output: dict[int, str] = {}
        ordered = sorted(indexes)
        for start in range(0, len(ordered), 800):
            chunk = ordered[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT id, value FROM strings WHERE id IN ({placeholders})", chunk
            )
            output.update({int(index): value for index, value in rows})
        return output

    def close(self) -> None:
        self.connection.close()
        self.temp_dir.cleanup()


def _column_index(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference)
    if not match:
        return 0
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _cell_token(cell: ET.Element) -> tuple[str, Any]:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "value", "".join(
            node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t")
        )
    value_node = cell.find(f"{{{MAIN_NS}}}v")
    value = value_node.text if value_node is not None and value_node.text is not None else ""
    if cell_type == "s" and value:
        return "shared", int(value)
    if cell_type == "b":
        return "value", "是" if value == "1" else "否"
    return "value", value


def _iter_sheet_rows(
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: SharedStringStore,
    block_size: int = 2000,
) -> Iterator[dict[int, str]]:
    pending: list[dict[int, tuple[str, Any]]] = []
    with archive.open(sheet_path) as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag != f"{{{MAIN_NS}}}row":
                continue
            row: dict[int, tuple[str, Any]] = {}
            for cell in element.findall(f"{{{MAIN_NS}}}c"):
                row[_column_index(cell.attrib.get("r", "A1"))] = _cell_token(cell)
            pending.append(row)
            element.clear()
            if len(pending) >= block_size:
                yield from _resolve_rows(pending, shared_strings)
                pending.clear()
    if pending:
        yield from _resolve_rows(pending, shared_strings)


def _resolve_rows(
    rows: list[dict[int, tuple[str, Any]]],
    shared_strings: SharedStringStore,
) -> Iterator[dict[int, str]]:
    indexes = {
        int(value)
        for row in rows
        for token_type, value in row.values()
        if token_type == "shared"
    }
    values = shared_strings.fetch_many(indexes)
    for row in rows:
        yield {
            column: values.get(int(value), "") if token_type == "shared" else str(value)
            for column, (token_type, value) in row.items()
        }


def _canonical_header(name: str) -> str | None:
    raw = name.strip()
    lowered = raw.lower()
    for canonical, aliases in COLUMN_ALIASES.items():
        if raw in aliases or lowered in aliases:
            return canonical
    return None


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def _parse_excel_date(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        serial = float(text)
        if 1 <= serial <= 100000:
            return datetime(1899, 12, 30) + timedelta(days=serial)
    except ValueError:
        pass
    normalized = text.replace("/", "-").replace("年", "-").replace("月", "-").replace("日", "")
    for parser in (datetime.fromisoformat,):
        try:
            return parser(normalized)
        except ValueError:
            continue
    for pattern in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y%m%d"):
        try:
            return datetime.strptime(normalized, pattern)
        except ValueError:
            continue
    return None


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "mean": round(statistics.fmean(ordered), 2),
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "max": float(ordered[-1]),
    }


def _source_category(source: str) -> str:
    for category, keywords in SOURCE_CATEGORY_KEYWORDS.items():
        if any(keyword in source for keyword in keywords):
            return category
    return "其他或内部来源"


def _profile_sheet(
    archive: zipfile.ZipFile,
    sheet: dict,
    shared_strings: SharedStringStore,
) -> tuple[dict, dict[str, set[str]], Counter[str]]:
    rows = _iter_sheet_rows(archive, sheet["path"], shared_strings)
    try:
        header_row = next(rows)
    except StopIteration:
        return (
            {"name": sheet["name"], "row_count": 0, "headers": []},
            defaultdict(set),
            Counter(),
        )

    max_column = max(header_row, default=-1)
    headers = [header_row.get(index, "").strip() for index in range(max_column + 1)]
    canonical_columns = {
        canonical: index
        for index, header in enumerate(headers)
        if (canonical := _canonical_header(header)) is not None
    }
    provenance_headers = [
        header
        for header in headers
        if any(keyword.lower() in header.lower() for keyword in PROVENANCE_HEADER_KEYWORDS)
    ]
    counters: dict[str, Counter[str]] = {
        "event_label": Counter(),
        "polarity_label": Counter(),
        "duplicate_flag": Counter(),
        "source_category": Counter(),
        "industry": Counter(),
        "duplication_id": Counter(),
    }
    missing = Counter()
    title_lengths: list[int] = []
    content_lengths: list[int] = []
    seen_ids: set[str] = set()
    duplicate_ids = 0
    title_hash_counts: Counter[str] = Counter()
    content_hash_counts: Counter[str] = Counter()
    sources: set[str] = set()
    entities: set[str] = set()
    min_date: datetime | None = None
    max_date: datetime | None = None
    invalid_dates = 0
    row_count = 0
    fingerprints: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        row_count += 1
        values = {
            canonical: row.get(column, "").strip()
            for canonical, column in canonical_columns.items()
        }
        for required in ("article_id", "title", "publish_time", "source", "content", "entity"):
            if required in canonical_columns and not values.get(required):
                missing[required] += 1

        article_id = values.get("article_id", "")
        if article_id:
            if article_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(article_id)
            fingerprints["article_id"].add(_hash_text(article_id))

        title = values.get("title", "")
        if title:
            title_hash = _hash_text(title)
            title_hash_counts[title_hash] += 1
            fingerprints["title"].add(title_hash)
            title_lengths.append(len(title))

        content = values.get("content", "")
        if content:
            content_hash = _hash_text(content)
            content_hash_counts[content_hash] += 1
            fingerprints["content"].add(content_hash)
            content_lengths.append(len(content))

        source = values.get("source", "")
        if source:
            sources.add(source)
            counters["source_category"][_source_category(source)] += 1

        entity = values.get("entity", "")
        if entity:
            entities.add(entity)
            fingerprints["entity"].add(_hash_text(entity))

        industry = values.get("industry", "")
        if industry:
            counters["industry"][industry] += 1

        for field in ("event_label", "polarity_label", "duplicate_flag"):
            if field in canonical_columns:
                counters[field][values.get(field, "") or "未标注"] += 1
        if "duplication_id" in canonical_columns:
            counters["duplication_id"][values.get("duplication_id", "") or "未分组"] += 1

        if "publish_time" in canonical_columns:
            date_value = values.get("publish_time", "")
            parsed = _parse_excel_date(date_value)
            if date_value and parsed is None:
                invalid_dates += 1
            if parsed:
                min_date = parsed if min_date is None or parsed < min_date else min_date
                max_date = parsed if max_date is None or parsed > max_date else max_date

    scope = "duplicate_supervision" if "重复" in sheet["name"] else "main_task"
    profile = {
        "name": sheet["name"],
        "scope": scope,
        "row_count": row_count,
        "headers": headers,
        "canonical_columns": canonical_columns,
        "provenance_headers": provenance_headers,
        "missing": dict(missing),
        "missing_rates": {
            field: round(count / row_count, 6) if row_count else 0.0
            for field, count in missing.items()
        },
        "duplicate_article_ids": duplicate_ids,
        "exact_duplicate_titles": sum(count - 1 for count in title_hash_counts.values() if count > 1),
        "exact_duplicate_contents": sum(count - 1 for count in content_hash_counts.values() if count > 1),
        "unique_source_count": len(sources),
        "unique_entity_count": len(entities),
        "source_category_distribution": dict(counters["source_category"].most_common()),
        "event_distribution": dict(counters["event_label"].most_common()),
        "polarity_distribution": dict(counters["polarity_label"].most_common()),
        "duplicate_flag_distribution": dict(counters["duplicate_flag"].most_common()),
        "duplication_group_count": sum(
            1 for group_id in counters["duplication_id"] if group_id != "未分组"
        ),
        "duplication_ungrouped_count": counters["duplication_id"].get("未分组", 0),
        "duplication_group_size": _percentiles(
            [
                count
                for group_id, count in counters["duplication_id"].items()
                if group_id != "未分组"
            ]
        ),
        "top_industries": dict(counters["industry"].most_common(20)),
        "title_length": _percentiles(title_lengths),
        "content_length": _percentiles(content_lengths),
        "date_min": min_date.date().isoformat() if min_date else None,
        "date_max": max_date.date().isoformat() if max_date else None,
        "invalid_date_count": invalid_dates,
    }
    duplication_groups = Counter(
        {
            group_id: count
            for group_id, count in counters["duplication_id"].items()
            if group_id != "未分组"
        }
    )
    return profile, fingerprints, duplication_groups


def _workbook_metadata(archive: zipfile.ZipFile) -> list[dict]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
    }
    sheets: list[dict] = []
    for node in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        relation_id = node.attrib[f"{{{REL_NS}}}id"]
        target = rel_map[relation_id].replace("\\", "/")
        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        sheets.append({"name": node.attrib.get("name", ""), "path": sheet_path})
    return sheets


def _full_workbook_profile(path: Path) -> tuple[dict, dict[str, dict[str, set[str]]]]:
    print(f"[audit] shared strings: {path.name}", flush=True)
    with zipfile.ZipFile(path) as archive:
        shared_strings = SharedStringStore(archive)
        try:
            profiles: list[dict] = []
            fingerprints: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
            combined_groups: dict[str, Counter[str]] = {
                "company": Counter(),
                "industry": Counter(),
            }
            for sheet in _workbook_metadata(archive):
                print(f"[audit] sheet: {path.name} / {sheet['name']}", flush=True)
                profile, sheet_fingerprints, duplication_groups = _profile_sheet(
                    archive, sheet, shared_strings
                )
                profiles.append(profile)
                scope = profile["scope"]
                for kind, values in sheet_fingerprints.items():
                    fingerprints[scope][kind].update(values)
                task_type = "industry" if "行业" in sheet["name"] else "company"
                combined_groups[task_type].update(duplication_groups)
            duplication_clusters = {
                task_type: _group_summary(groups)
                for task_type, groups in combined_groups.items()
                if groups
            }
            return {
                "file": path.name,
                "sheets": profiles,
                "duplication_clusters": duplication_clusters,
            }, fingerprints
        finally:
            shared_strings.close()


def _group_summary(groups: Counter[str]) -> dict:
    sizes = list(groups.values())
    return {
        "group_count": len(groups),
        "grouped_article_count": sum(sizes),
        "positive_group_count": sum(size >= 2 for size in sizes),
        "singleton_group_count": sum(size == 1 for size in sizes),
        "group_size": _percentiles(sizes),
    }


def _json_shape(value: Any, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(key): _json_shape(child, depth + 1) for key, child in list(value.items())[:30]}
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "item_shape": _json_shape(value[0], depth + 1) if value else None,
        }
    return type(value).__name__


def _full_json_profile(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    event_names: set[str] = set()
    schema_lengths: list[int] = []
    key_counter: Counter[str] = Counter()

    def visit(value: Any, parent_key: str = "") -> None:
        if isinstance(value, dict):
            key_counter.update(str(key) for key in value)
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in {"event", "event_type", "event_name", "事件", "事件类型", "事件名称", "name"}:
                    if isinstance(child, str) and len(child) <= 100:
                        event_names.add(child.strip())
                visit(child, str(key))
        elif isinstance(value, list):
            if parent_key == "event_schema":
                schema_lengths.append(len(value))
            for child in value:
                visit(child, parent_key)

    visit(payload)
    return {
        "file": path.name,
        "top_level_count": len(payload) if isinstance(payload, (list, dict)) else 1,
        "shape": _json_shape(payload),
        "common_keys": dict(key_counter.most_common(30)),
        "event_name_count": len(event_names),
        "event_names": sorted(name for name in event_names if name),
        "event_schema_length": _percentiles(schema_lengths),
    }


def _overlap(left: set[str], right: set[str]) -> dict[str, float | int]:
    intersection = len(left & right)
    denominator = min(len(left), len(right))
    return {
        "intersection": intersection,
        "smaller_set_overlap_rate": round(intersection / denominator, 6) if denominator else 0.0,
    }


def _pairwise_overlaps(
    fingerprints: dict[str, dict[str, dict[str, set[str]]]],
) -> list[dict]:
    pairs = [
        ("news_with_tags_train.xlsx", "news_with_tags_test.xlsx", "main_task"),
        ("news_without_tags_train.xlsx", "news_without_tags_test.xlsx", "main_task"),
        ("news_with_tags_train.xlsx", "news_without_tags_train.xlsx", "main_task"),
        ("news_with_tags_test.xlsx", "news_without_tags_test.xlsx", "main_task"),
    ]
    output: list[dict] = []
    for left_name, right_name, scope in pairs:
        left = fingerprints.get(left_name, {}).get(scope, {})
        right = fingerprints.get(right_name, {}).get(scope, {})
        output.append(
            {
                "left": left_name,
                "right": right_name,
                "scope": scope,
                "article_id": _overlap(left.get("article_id", set()), right.get("article_id", set())),
                "title": _overlap(left.get("title", set()), right.get("title", set())),
                "content": _overlap(left.get("content", set()), right.get("content", set())),
                "entity": _overlap(left.get("entity", set()), right.get("entity", set())),
            }
        )
    return output


def _find_profile(
    workbooks: list[dict], file_name: str, sheet_name: str
) -> dict:
    for workbook in workbooks:
        if workbook["file"] != file_name:
            continue
        for sheet in workbook["sheets"]:
            if sheet["name"] == sheet_name:
                return sheet
    raise KeyError(f"未找到 {file_name}/{sheet_name}")


def _label_alignment(workbooks: list[dict], event_schemas: list[dict]) -> dict:
    schema_map = {row["file"]: row for row in event_schemas}
    tasks = {
        "company": (
            _find_profile(workbooks, "news_with_tags_train.xlsx", "个股新闻"),
            _find_profile(workbooks, "news_with_tags_test.xlsx", "个股新闻"),
            schema_map["事件类型_标的.json"],
        ),
        "industry": (
            _find_profile(workbooks, "news_with_tags_train.xlsx", "行业新闻"),
            _find_profile(workbooks, "news_with_tags_test.xlsx", "行业新闻"),
            schema_map["事件类型_行业.json"],
        ),
    }
    output: dict[str, dict] = {}
    for task_name, (train, test, event_schema) in tasks.items():
        train_labels = set(train["event_distribution"])
        test_labels = set(test["event_distribution"])
        schema_labels = set(event_schema["event_names"])
        train_counts = list(train["event_distribution"].values())
        output[task_name] = {
            "train_class_count": len(train_labels),
            "test_class_count": len(test_labels),
            "test_unseen_in_train": sorted(test_labels - train_labels),
            "train_labels_not_in_schema": sorted(train_labels - schema_labels),
            "schema_labels_absent_from_train": sorted(schema_labels - train_labels),
            "schema_coverage_train": round(
                len(train_labels & schema_labels) / len(train_labels), 6
            )
            if train_labels
            else 0.0,
            "schema_coverage_test": round(
                len(test_labels & schema_labels) / len(test_labels), 6
            )
            if test_labels
            else 0.0,
            "train_classes_lt_5": sum(count < 5 for count in train_counts),
            "train_classes_lt_10": sum(count < 10 for count in train_counts),
        }
    return output


def _render_distribution(distribution: dict[str, int], limit: int = 30) -> list[str]:
    return [f"  - {label or '空值'}：{count}" for label, count in list(distribution.items())[:limit]]


def render_full_markdown(payload: dict) -> str:
    lines = [
        "# EventLens 真实脱敏数据审计",
        "",
        "## 审计边界",
        "",
        "- 只输出聚合统计、字段模式和不可逆哈希重叠结果。",
        "- 不在报告中展示内部业务原文、公司样本或来源明细。",
        "- 原始文件保持只读，未进行覆盖或格式转换。",
        "",
        "## 工作簿画像",
        "",
    ]
    for workbook in payload["workbooks"]:
        lines.extend([f"### `{workbook['file']}`", ""])
        if workbook.get("duplication_clusters"):
            lines.append(f"- 跨工作表 duplication_id 聚合：{workbook['duplication_clusters']}")
            lines.append("")
        for sheet in workbook["sheets"]:
            lines.extend(
                [
                    f"#### `{sheet['name']}`",
                    "",
                    f"- 任务范围：{sheet['scope']}",
                    f"- 数据行数：{sheet['row_count']}",
                    f"- 原始字段：{', '.join(sheet['headers'])}",
                    f"- 可识别标准字段：{', '.join(sheet['canonical_columns'])}",
                    f"- 显式公开/内部溯源字段：{', '.join(sheet['provenance_headers']) or '无'}",
                    f"- 重复 article_id：{sheet['duplicate_article_ids']}",
                    f"- 标题精确重复行：{sheet['exact_duplicate_titles']}",
                    f"- 正文精确重复行：{sheet['exact_duplicate_contents']}",
                    f"- 重复组数量：{sheet['duplication_group_count']}",
                    f"- 未分配重复组：{sheet['duplication_ungrouped_count']}",
                    f"- 重复组规模：{sheet['duplication_group_size']}",
                    f"- 唯一来源数：{sheet['unique_source_count']}",
                    f"- 唯一实体数：{sheet['unique_entity_count']}",
                    f"- 日期范围：{sheet['date_min']} 至 {sheet['date_max']}",
                    f"- 无法解析日期：{sheet['invalid_date_count']}",
                    f"- 标题长度：{sheet['title_length']}",
                    f"- 正文长度：{sheet['content_length']}",
                ]
            )
            if sheet["missing_rates"]:
                lines.append(f"- 缺失率：{sheet['missing_rates']}")
            lines.extend(["- 来源类别：", *_render_distribution(sheet["source_category_distribution"])])
            if sheet["event_distribution"]:
                lines.extend(["- 事件标签：", *_render_distribution(sheet["event_distribution"])])
            if sheet["polarity_distribution"]:
                lines.extend(["- 情感标签：", *_render_distribution(sheet["polarity_distribution"])])
            if sheet["duplicate_flag_distribution"]:
                lines.extend(["- 重复标签：", *_render_distribution(sheet["duplicate_flag_distribution"])])
            lines.append("")

    lines.extend(["## 跨集合精确重叠", ""])
    for overlap in payload["overlaps"]:
        lines.extend(
            [
                f"### `{overlap['left']}` ↔ `{overlap['right']}`",
                "",
                f"- article_id：{overlap['article_id']}",
                f"- 标题哈希：{overlap['title']}",
                f"- 正文哈希：{overlap['content']}",
                f"- 实体哈希：{overlap['entity']}",
                "",
            ]
        )

    lines.extend(["## 事件体系 JSON", ""])
    for item in payload["event_schemas"]:
        lines.extend(
            [
                f"### `{item['file']}`",
                "",
                f"- 顶层条目数：{item['top_level_count']}",
                f"- 识别到的事件名称数：{item['event_name_count']}",
                f"- event_schema 长度：{item['event_schema_length']}",
                f"- 常见结构字段：{item['common_keys']}",
                "",
            ]
        )
    lines.extend(["## 标签空间对齐", ""])
    for task_name, alignment in payload["label_alignment"].items():
        lines.extend(
            [
                f"### `{task_name}`",
                "",
                f"- 训练类别数：{alignment['train_class_count']}",
                f"- 测试类别数：{alignment['test_class_count']}",
                f"- 测试新类：{alignment['test_unseen_in_train'] or '无'}",
                f"- 事件体系训练覆盖率：{alignment['schema_coverage_train']:.2%}",
                f"- 事件体系测试覆盖率：{alignment['schema_coverage_test']:.2%}",
                f"- 训练中少于 5 条的类别：{alignment['train_classes_lt_5']}",
                f"- 训练中少于 10 条的类别：{alignment['train_classes_lt_10']}",
                f"- 训练标签未进入事件体系：{alignment['train_labels_not_in_schema'] or '无'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _json_structure(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    result = {
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "top_level_type": type(payload).__name__,
    }
    if isinstance(payload, list):
        result["item_count"] = len(payload)
        result["sample_item_types"] = sorted({type(item).__name__ for item in payload[:100]})
        if payload and isinstance(payload[0], dict):
            result["sample_keys"] = sorted(
                {str(key) for item in payload[:100] if isinstance(item, dict) for key in item}
            )
    elif isinstance(payload, dict):
        result["key_count"] = len(payload)
        result["keys"] = sorted(str(key) for key in payload)[:100]
        value_types = {type(value).__name__ for value in payload.values()}
        result["value_types"] = sorted(value_types)
    return result


def _dimension_rows(dimension: str) -> int | None:
    if not dimension:
        return None
    match = re.search(r"(\d+)$", dimension)
    return int(match.group(1)) if match else None


def render_markdown(payload: dict) -> str:
    lines = [
        "# 原始数据结构审计",
        "",
        "本报告只读取文件结构和模式，不输出内部业务原文或可识别记录。",
        "",
        "## XLSX 文件",
        "",
    ]
    for workbook in payload["xlsx"]:
        lines.extend(
            [
                f"### `{workbook['file']}`",
                "",
                f"- 文件大小：{workbook['file_bytes'] / 1024 / 1024:.2f} MB",
                f"- Shared Strings 解压大小：{workbook['shared_strings_raw_bytes'] / 1024 / 1024:.2f} MB",
            ]
        )
        for sheet in workbook["sheets"]:
            rows = _dimension_rows(sheet["dimension"])
            data_rows = max(0, rows - 1) if rows is not None else "未知"
            lines.append(
                f"- 工作表 `{sheet['name']}`：范围 `{sheet['dimension']}`，约 {data_rows} 条数据"
            )
        lines.append("")
    lines.extend(["## JSON 文件", ""])
    for item in payload["json"]:
        lines.extend(
            [
                f"### `{item['file']}`",
                "",
                f"- 顶层类型：{item['top_level_type']}",
                f"- 文件大小：{item['file_bytes'] / 1024:.2f} KB",
            ]
        )
        if "item_count" in item:
            lines.append(f"- 条目数：{item['item_count']}")
        if "key_count" in item:
            lines.append(f"- 顶层键数量：{item['key_count']}")
        if item.get("sample_keys"):
            lines.append(f"- 样例字段：{', '.join(item['sample_keys'])}")
        if item.get("keys"):
            lines.append(f"- 顶层键：{', '.join(item['keys'])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="只读审计 EventLens 原始数据结构")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--json-output", default="reports/raw_data_structure.json")
    parser.add_argument("--markdown-output", default="reports/raw_data_structure.md")
    parser.add_argument(
        "--full",
        action="store_true",
        help="流式扫描全部数据，生成缺失、标签、重复和跨集合重叠统计",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    if args.full:
        workbook_profiles: list[dict] = []
        workbook_fingerprints: dict[str, dict[str, dict[str, set[str]]]] = {}
        for path in sorted(root.glob("*.xlsx")):
            profile, fingerprints = _full_workbook_profile(path)
            workbook_profiles.append(profile)
            workbook_fingerprints[path.name] = fingerprints
        event_schemas = [_full_json_profile(path) for path in sorted(root.glob("*.json"))]
        payload = {
            "workbooks": workbook_profiles,
            "overlaps": _pairwise_overlaps(workbook_fingerprints),
            "event_schemas": event_schemas,
            "label_alignment": _label_alignment(workbook_profiles, event_schemas),
        }
        json_path = Path(args.json_output)
        markdown_path = Path(args.markdown_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(render_full_markdown(payload), encoding="utf-8")
        print(markdown_path.as_posix())
        return

    payload = {
        "xlsx": [_xlsx_structure(path) for path in sorted(root.glob("*.xlsx"))],
        "json": [_json_structure(path) for path in sorted(root.glob("*.json"))],
    }
    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(markdown_path.as_posix())


if __name__ == "__main__":
    main()
