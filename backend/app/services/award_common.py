from typing import Any

import pandas as pd


FIELD_LABELS = {
    "student_id": "学号",
    "name": "姓名",
    "college": "书院",
    "department": "学院",
    "award_level": "奖项等级",
    "amount": "奖励金额",
    "organizer": "举办单位",
}

LEVEL_TIERS = {
    "国家": 4,
    "省": 3,
    "市": 2,
    "校": 1,
}

LEVEL_GRADES = {
    "A": 3,
    "B": 2,
    "C": 1,
}

ACTION_LABELS = {
    "apply": "已申请/已申报/已提交",
    "approve": "已审批/已审核/已通过",
    "publish": "已公示/已公布",
}

ACTION_KEYWORDS = {
    "apply": ["申请", "申报", "提交"],
    "approve": ["审批", "审核", "通过"],
    "publish": ["公示", "公布"],
}


def _field_column(fields: dict[str, dict[str, Any]], field_key: str) -> str | None:
    field = fields.get(field_key) or {}
    return field.get("column")

def _numeric_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if not column:
        return pd.Series(dtype="float64")

    values = (
        frame[column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("元", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(values, errors="coerce")

def _row_numbers(frame: pd.DataFrame) -> list[int]:
    return [int(index) + 2 for index in frame.index[:10]]

def _round_number(value: Any) -> float:
    return round(float(value), 2)

def _to_plain_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value

def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)

def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _to_plain_value(value) for key, value in row.items()}
        for row in frame.where(pd.notna(frame), None).to_dict(orient="records")
    ]

def _clean_optional_text(value: Any) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in ["nan", "none", "null"]:
        return None

    return text

def _drop_rows_without_student_identity(frame: pd.DataFrame, identity_columns: list[str]) -> pd.DataFrame:
    if not identity_columns:
        return frame

    keep_mask = pd.Series(False, index=frame.index)
    for column in identity_columns:
        values = frame[column].apply(_clean_optional_text)
        keep_mask = keep_mask | values.notna()

    return frame[keep_mask]

def _extract_existing_value(frame: pd.DataFrame, column: str, question: str) -> str | None:
    """从问题中找出表格里真实存在的值。

    这样可以避免乱猜人名、书院名或学号。
    """

    values = (
        frame[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    candidates = sorted(values.unique().tolist(), key=len, reverse=True)

    for value in candidates:
        if value and value in question:
            return value

    return None

def _filter_by_text_value(frame: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    values = frame[column].fillna("").astype(str).str.strip()
    return frame[values == target]

def _build_entity_award_result(
    matched: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    target: str,
    query_type: str,
    amount_column: str | None,
) -> dict[str, Any]:
    amount_values = _numeric_series(matched, amount_column)
    total_amount = amount_values.sum() if amount_column else 0

    return {
        "found": not matched.empty,
        "query_type": query_type,
        "target": target,
        "record_count": int(len(matched)),
        "total_amount": _round_number(total_amount),
        "fields": fields,
        "records": _records_from_frame(matched.head(10)),
        "message": "" if not matched.empty else f"没有找到“{target}”的获奖记录",
    }

def _build_group_award_result(
    matched: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    target: str,
    query_type: str,
    amount_column: str | None,
) -> dict[str, Any]:
    amount_values = _numeric_series(matched, amount_column)
    total_amount = amount_values.sum() if amount_column else 0
    name_column = _field_column(fields, "name")
    level_column = _field_column(fields, "award_level")

    unique_people = 0
    if name_column and name_column in matched.columns:
        unique_people = int(matched[name_column].dropna().astype(str).str.strip().nunique())

    return {
        "found": not matched.empty,
        "query_type": query_type,
        "target": target,
        "record_count": int(len(matched)),
        "student_count": unique_people,
        "total_amount": _round_number(total_amount),
        "award_level_distribution": _value_counts(matched, level_column),
        "fields": fields,
        "records": _records_from_frame(matched.head(10)),
        "message": "" if not matched.empty else f"没有找到“{target}”的获奖记录",
    }

def _value_counts(frame: pd.DataFrame, column: str | None) -> dict[str, int]:
    if not column:
        return {}

    counts = frame[column].dropna().astype(str).value_counts().head(10)
    return {str(key): int(value) for key, value in counts.items()}

def _not_found_result(message: str) -> dict[str, Any]:
    return {
        "found": False,
        "message": message,
        "record_count": 0,
        "total_amount": 0,
        "records": [],
    }

def _find_column_by_keywords(frame: pd.DataFrame, keywords: list[str]) -> str | None:
    for column in frame.columns:
        column_text = str(column)
        if any(keyword in column_text for keyword in keywords):
            return str(column)
    return None

def _unsupported_award_answer(fields: dict[str, dict[str, Any]], message: str) -> dict[str, Any]:
    return {
        "answered": False,
        "query_type": "unsupported",
        "message": message,
        "fields": fields,
        "rows": [],
    }
