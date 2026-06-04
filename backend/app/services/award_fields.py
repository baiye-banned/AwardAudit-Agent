import json
from typing import Any

import httpx
import pandas as pd

from app.services.award_common import FIELD_LABELS, _numeric_series
from app.services.llm_config import get_llm_config


FIELD_KEYWORDS = {
    "student_id": ["学号", "学生编号", "学生号", "student id", "id"],
    "name": ["姓名", "学生姓名", "名字", "name"],
    "college": ["住宿书院", "宿舍书院", "所在书院", "归属书院", "书院"],
    "department": ["专业学院", "所在学院", "学院", "院系", "学部", "department"],
    "award_level": ["获奖等级", "获奖级别", "奖项等级", "奖项级别", "收录情况", "级别"],
    "amount": ["奖励额度", "奖励金额", "获奖额度", "获奖金额", "金额", "额度"],
    "organizer": ["举办单位", "主办单位", "期刊名", "组织单位", "单位"],
}


def infer_award_fields(frame: pd.DataFrame, use_llm: bool = True) -> dict[str, dict[str, Any]]:
    """识别奖项名单里的关键业务字段。

    LLM 只负责给出候选列名；后端会继续校验列是否存在、数据是否可用。
    没有 API Key 或 LLM 调用失败时，使用关键词 fallback，保证演示稳定。
    """

    llm_candidates = _infer_fields_with_llm(frame) if use_llm else {}
    fallback_candidates = _infer_fields_by_keywords(frame)
    result = {}

    for field_key in FIELD_LABELS:
        candidate = llm_candidates.get(field_key) or fallback_candidates.get(field_key)
        source = "llm" if llm_candidates.get(field_key) else "fallback"
        result[field_key] = _validate_field(frame, field_key, candidate, source)

    return result

def _infer_fields_with_llm(frame: pd.DataFrame) -> dict[str, str]:
    llm_config = get_llm_config()
    if not llm_config.api_key:
        return {}

    prompt = _build_field_prompt(frame)

    try:
        response = httpx.post(
            f"{llm_config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {llm_config.api_key}"},
            json={
                "model": llm_config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你只返回 JSON，不要输出解释文字。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=20,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_llm_json(content)
    except Exception:
        return {}

def _build_field_prompt(frame: pd.DataFrame) -> str:
    profile = {
        "columns": [str(column) for column in frame.columns],
        "sample_rows": frame.head(5).where(pd.notna(frame), None).to_dict(orient="records"),
        "non_empty_rate": {
            str(column): round(float(frame[column].notna().mean()), 3)
            for column in frame.columns
        },
    }
    return (
        "请从高校奖项名单中识别字段。字段包括："
        "student_id、name、college、department、award_level、amount、organizer。\n"
        "请返回 JSON，key 是字段名，value 是原始列名；无法判断则 value 为 null。\n"
        f"表格信息：{json.dumps(profile, ensure_ascii=False)}"
    )

def _parse_llm_json(content: str) -> dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()

    data = json.loads(text)
    return {
        key: value
        for key, value in data.items()
        if key in FIELD_LABELS and isinstance(value, str) and value
    }

def _infer_fields_by_keywords(frame: pd.DataFrame) -> dict[str, str]:
    columns = [str(column) for column in frame.columns]
    result = {}

    for field_key, keywords in FIELD_KEYWORDS.items():
        for keyword in keywords:
            match = _find_column_by_keyword(columns, keyword)
            if match:
                result[field_key] = match
                break

    return result

def _find_column_by_keyword(columns: list[str], keyword: str) -> str | None:
    keyword_lower = keyword.lower()
    for column in columns:
        if keyword_lower in column.lower():
            return column
    return None

def _validate_field(
    frame: pd.DataFrame,
    field_key: str,
    column: str | None,
    source: str,
) -> dict[str, Any]:
    if not column or column not in frame.columns:
        return {
            "column": None,
            "confidence": "low",
            "reason": "未识别到可用字段",
        }

    non_empty_rate = float(frame[column].notna().mean())
    if non_empty_rate < 0.2:
        return {
            "column": None,
            "confidence": "low",
            "reason": f"候选列 {column} 非空率过低",
        }

    if field_key == "amount":
        numeric_rate = float(_numeric_series(frame, column).notna().mean())
        if numeric_rate < 0.5:
            return {
                "column": None,
                "confidence": "low",
                "reason": f"候选列 {column} 无法稳定转换为数字",
            }

    confidence = "high" if source == "llm" else "medium"
    return {
        "column": column,
        "confidence": confidence,
        "reason": "LLM 候选通过程序校验" if source == "llm" else "规则 fallback 命中并通过校验",
    }
