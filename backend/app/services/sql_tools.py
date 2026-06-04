import os
import re
import sqlite3
from typing import Any

import httpx
import pandas as pd

from app.services.award_tools import infer_award_fields


TABLE_NAME = "awards"
DEFAULT_LIMIT = 50
DANGEROUS_SQL_WORDS = [
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
]


def answer_by_sql(frame: pd.DataFrame, question: str) -> dict[str, Any]:
    """用临时 SQLite 执行只读 SQL 查询。

    LLM 只负责生成 SQL；后端必须校验 SQL，再用 SQLite 执行。
    没有 API Key 时使用 fallback，保证本地演示不会崩。
    """

    fields = infer_award_fields(frame)
    sql = generate_sql_with_llm(frame, question, fields) or generate_fallback_sql(frame, question)

    if not sql:
        return {
            "success": False,
            "message": "暂时无法为这个问题生成可靠 SQL",
            "sql": "",
            "rows": [],
        }

    try:
        return run_select_sql(frame, sql)
    except Exception as exc:
        return {
            "success": False,
            "message": str(exc),
            "sql": sql,
            "rows": [],
        }


def generate_sql_with_llm(
    frame: pd.DataFrame,
    question: str,
    fields: dict[str, dict[str, Any]],
) -> str | None:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return None

    prompt = _build_sql_prompt(frame, question, fields)
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你只返回一条 SQLite SELECT SQL，不要解释，不要 Markdown。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=20,
        )
        response.raise_for_status()
        return _strip_sql_text(response.json()["choices"][0]["message"]["content"])
    except Exception:
        return None


def generate_fallback_sql(frame: pd.DataFrame, question: str) -> str | None:
    """没有 LLM 时的兜底 SQL。

    只覆盖最常见的奖项金额排行问题，复杂开放问题交给 LLM。
    """

    plan = parse_query_plan(frame, question)
    return build_sql_from_plan(frame, plan)


def parse_query_plan(frame: pd.DataFrame, question: str) -> dict[str, Any] | None:
    """把常见中文问题解析成稳定的查询计划。

    QueryPlan 是确定性结构；同样问题和同样字段会得到同样 plan。
    """

    if not _looks_like_ranking_question(question):
        return None

    return {
        "intent": "rank_students",
        "entity": _parse_rank_entity(question),
        "metric": _parse_rank_metric(question),
        "rank_index": _parse_rank_index(question),
        "top_n": _parse_top_n(question),
    }


def build_sql_from_plan(frame: pd.DataFrame, plan: dict[str, Any] | None) -> str | None:
    """根据 QueryPlan 生成固定 SQL 模板。"""

    if not plan:
        return None

    if plan.get("intent") not in ["rank_students", "rank_students_by_amount"]:
        return None

    fields = infer_award_fields(frame, use_llm=False)
    entity = plan.get("entity") or "student"
    entity_column = _entity_column(fields, entity)
    amount_column = _field_column(fields, "amount")

    if not entity_column:
        return None

    metric = plan.get("metric") or "sum_amount"
    if metric == "sum_amount" and not amount_column:
        return None

    top_n = plan.get("top_n")
    if top_n:
        limit = int(top_n)
        offset = 0
    else:
        limit = 1
        offset = max(int(plan.get("rank_index", 1)) - 1, 0)

    metric_sql = 'COUNT(*) AS award_count' if metric == "award_count" else f'SUM("{amount_column}") AS total_amount'
    order_column = "award_count" if metric == "award_count" else "total_amount"

    return (
        f'SELECT "{entity_column}", {metric_sql}, COUNT(*) AS record_count '
        f'FROM {TABLE_NAME} '
        f'WHERE "{entity_column}" IS NOT NULL AND TRIM(CAST("{entity_column}" AS TEXT)) != "" '
        f'GROUP BY "{entity_column}" '
        f'ORDER BY {order_column} DESC, "{entity_column}" ASC '
        f'LIMIT {limit} OFFSET {offset}'
    )


def validate_select_sql(sql: str) -> str:
    cleaned = _strip_sql_text(sql)
    lowered = cleaned.lower()

    if ";" in cleaned.rstrip(";"):
        raise ValueError("SQL 只允许一条语句")

    if not (lowered.startswith("select ") or lowered.startswith("with ")):
        raise ValueError("只允许执行 SELECT 查询")

    for word in DANGEROUS_SQL_WORDS:
        if re.search(rf"\b{word}\b", lowered):
            raise ValueError(f"SQL 包含禁止关键词：{word}")

    if TABLE_NAME not in lowered:
        raise ValueError(f"SQL 只能查询 {TABLE_NAME} 表")

    if " limit " not in f" {lowered} ":
        cleaned = f"{cleaned.rstrip(';')} LIMIT {DEFAULT_LIMIT}"

    return cleaned.rstrip(";")


def run_select_sql(frame: pd.DataFrame, sql: str) -> dict[str, Any]:
    safe_sql = validate_select_sql(sql)
    safe_frame = _prepare_sql_frame(frame)

    with sqlite3.connect(":memory:") as connection:
        safe_frame.to_sql(TABLE_NAME, connection, index=False, if_exists="replace")
        result_frame = pd.read_sql_query(safe_sql, connection)

    return {
        "success": True,
        "sql": safe_sql,
        "rows": _safe_records(result_frame),
        "row_count": int(len(result_frame)),
    }


def _build_sql_prompt(
    frame: pd.DataFrame,
    question: str,
    fields: dict[str, dict[str, Any]],
) -> str:
    schema = {
        "table": TABLE_NAME,
        "columns": [str(column) for column in frame.columns],
        "award_fields": fields,
        "sample_rows": frame.head(3).where(pd.notna(frame), None).to_dict(orient="records"),
    }

    return (
        "请根据用户问题生成 SQLite SQL。\n"
        "硬性要求：只能查询 awards 表；只能生成 SELECT 或 WITH；必须包含 LIMIT；中文列名请用双引号包裹。\n"
        f"用户问题：{question}\n"
        f"表结构：{schema}"
    )


def _prepare_sql_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column) for column in result.columns]
    return result


def _strip_sql_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        cleaned = cleaned.removeprefix("sql").strip()
    return cleaned.strip()


def _looks_like_ranking_question(question: str) -> bool:
    return any(word in question for word in ["获奖", "钱", "金额", "奖励", "额度"]) and any(
        word in question
        for word in ["最多", "最高", "第一", "第二", "第三", "第1", "第2", "第3", "top"]
    )


def _parse_rank_metric(question: str) -> str:
    if any(word in question for word in ["钱", "金额", "额度", "总额"]):
        return "sum_amount"
    return "award_count"


def _parse_rank_entity(question: str) -> str:
    if "书院" in question:
        return "college"
    if "学院" in question:
        return "department"
    return "student"


def _entity_column(fields: dict[str, dict[str, Any]], entity: str) -> str | None:
    if entity == "college":
        return _field_column(fields, "college")
    if entity == "department":
        return _field_column(fields, "department")
    return _field_column(fields, "name")


def _parse_rank_index(question: str) -> int:
    rank_words = {
        "第一": 1,
        "最高": 1,
        "最多": 1,
        "第二": 2,
        "第三": 3,
        "第四": 4,
        "第五": 5,
    }

    for word, value in rank_words.items():
        if word in question:
            return value

    match = re.search(r"第\s*(\d+)", question)
    if match:
        return int(match.group(1))

    return 1


def _parse_top_n(question: str) -> int | None:
    top_words = {
        "前一": 1,
        "前二": 2,
        "前三": 3,
        "前四": 4,
        "前五": 5,
        "前十": 10,
    }

    for word, value in top_words.items():
        if word in question:
            return value

    match = re.search(r"(?:top|前)\s*(\d+)", question.lower())
    if match:
        return int(match.group(1))

    return None


def _field_column(fields: dict[str, dict[str, Any]], key: str) -> str | None:
    field = fields.get(key) or {}
    column = field.get("column")
    if isinstance(column, str) and column:
        return column
    return None


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    return [
        {str(key): _to_plain_value(value) for key, value in row.items()}
        for row in records
    ]


def _to_plain_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
