from typing import Any

import pandas as pd

from app.services.award_common import FIELD_LABELS, _field_column, _numeric_series, _round_number, _row_numbers
from app.services.award_fields import infer_award_fields


def quality_check_awards(frame: pd.DataFrame, use_llm: bool = True) -> dict[str, Any]:
    fields = infer_award_fields(frame, use_llm=use_llm)
    issues = []

    _add_missing_value_issue(frame, fields, issues, "student_id", "missing_student_id")
    _add_missing_value_issue(frame, fields, issues, "name", "missing_name")
    _add_missing_value_issue(frame, fields, issues, "college", "missing_college")
    _add_missing_value_issue(frame, fields, issues, "amount", "missing_amount")
    _add_duplicate_student_issue(frame, fields, issues)
    _add_amount_outlier_issue(frame, fields, issues)

    return {
        "fields": fields,
        "quality_issues": issues,
        "issue_count": len(issues),
    }

def summarize_awards(frame: pd.DataFrame, use_llm: bool = True) -> dict[str, Any]:
    fields = infer_award_fields(frame, use_llm=use_llm)
    amount_column = _field_column(fields, "amount")
    college_column = _field_column(fields, "college")
    department_column = _field_column(fields, "department")
    level_column = _field_column(fields, "award_level")

    amount_values = _numeric_series(frame, amount_column)
    total_amount = float(amount_values.sum()) if amount_column else 0.0
    valid_amount_count = int(amount_values.notna().sum()) if amount_column else 0

    return {
        "fields": fields,
        "total_rows": int(len(frame)),
        "total_amount": _round_number(total_amount),
        "average_amount": _round_number(total_amount / valid_amount_count) if valid_amount_count else 0,
        "college_amount_top": _group_sum(frame, college_column, amount_column),
        "department_count_top": _group_count(frame, department_column),
        "award_level_distribution": _value_counts(frame, level_column),
    }

def build_audit_report(frame: pd.DataFrame, use_llm: bool = True) -> dict[str, Any]:
    fields = infer_award_fields(frame, use_llm=use_llm)
    quality = quality_check_awards(frame, use_llm=use_llm)
    summary = summarize_awards(frame, use_llm=use_llm)
    report = _build_report_text(fields, quality["quality_issues"], summary)

    return {
        "fields": fields,
        "quality_issues": quality["quality_issues"],
        "summary": summary,
        "report": report,
    }

def _add_missing_value_issue(
    frame: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
    field_key: str,
    issue_type: str,
) -> None:
    column = _field_column(fields, field_key)
    if not column:
        return

    missing_rows = frame[frame[column].isna() | (frame[column].astype(str).str.strip() == "")]
    if missing_rows.empty:
        return

    issues.append(
        {
            "type": issue_type,
            "field": field_key,
            "column": column,
            "count": int(len(missing_rows)),
            "rows": _row_numbers(missing_rows),
            "message": f"{FIELD_LABELS[field_key]}存在缺失值",
        }
    )

def _add_duplicate_student_issue(
    frame: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    column = _field_column(fields, "student_id")
    if not column:
        return

    values = frame[column].astype(str).str.strip()
    values = values[values != ""]
    duplicated_values = values[values.duplicated()].unique().tolist()
    if not duplicated_values:
        return

    issues.append(
        {
            "type": "duplicate_student_id",
            "field": "student_id",
            "column": column,
            "count": len(duplicated_values),
            "values": duplicated_values[:10],
            "message": "发现重复学号，可能存在同一学生多次记录",
        }
    )

def _add_amount_outlier_issue(
    frame: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
) -> None:
    column = _field_column(fields, "amount")
    if not column:
        return

    values = _numeric_series(frame, column)
    valid_values = values.dropna()
    if valid_values.empty:
        return

    q1 = valid_values.quantile(0.25)
    q3 = valid_values.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    median = valid_values.median()
    if len(valid_values) < 8:
        business_upper = max(median * 5, 1000)
    else:
        business_upper = max(min(upper, median * 5), 1000)
    outlier_rows = frame[values > business_upper]

    if outlier_rows.empty:
        return

    issues.append(
        {
            "type": "amount_outlier",
            "field": "amount",
            "column": column,
            "count": int(len(outlier_rows)),
            "rows": _row_numbers(outlier_rows),
            "message": "发现奖励金额明显偏高的记录",
        }
    )

def _group_sum(frame: pd.DataFrame, group_column: str | None, value_column: str | None) -> list[dict[str, Any]]:
    if not group_column or not value_column:
        return []

    temp = frame[[group_column]].copy()
    temp[value_column] = _numeric_series(frame, value_column)
    temp = temp.dropna(subset=[group_column, value_column])
    grouped = temp.groupby(group_column)[value_column].sum().sort_values(ascending=False).head(10)

    return [
        {"group": str(group), "sum": _round_number(value)}
        for group, value in grouped.items()
    ]

def _group_count(frame: pd.DataFrame, group_column: str | None) -> list[dict[str, Any]]:
    if not group_column:
        return []

    counts = frame[group_column].dropna().astype(str).value_counts().head(10)
    return [
        {"group": str(group), "count": int(count)}
        for group, count in counts.items()
    ]

def _value_counts(frame: pd.DataFrame, column: str | None) -> dict[str, int]:
    if not column:
        return {}

    counts = frame[column].dropna().astype(str).value_counts().head(10)
    return {str(key): int(value) for key, value in counts.items()}

def _build_report_text(
    fields: dict[str, dict[str, Any]],
    issues: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    field_lines = [
        f"- {label}：{fields[key]['column'] or '未识别'}（{fields[key]['confidence']}）"
        for key, label in FIELD_LABELS.items()
    ]
    issue_lines = [f"- {issue['message']}：{issue['count']} 条" for issue in issues[:8]]
    issue_text = "\n".join(issue_lines) if issue_lines else "- 暂未发现明显审核问题"

    top_college = summary["college_amount_top"][0] if summary["college_amount_top"] else None
    top_college_text = (
        f"{top_college['group']}，合计 {top_college['sum']}"
        if top_college
        else "暂无可统计数据"
    )

    return "\n".join(
        [
            "审核报告",
            "",
            "一、字段识别",
            *field_lines,
            "",
            "二、关键统计",
            f"- 总记录数：{summary['total_rows']}",
            f"- 总奖励金额：{summary['total_amount']}",
            f"- 人均奖励金额：{summary['average_amount']}",
            f"- 奖励金额最高书院：{top_college_text}",
            "",
            "三、审核问题",
            issue_text,
            "",
            "四、审核建议",
            "- 请优先核对缺失字段、重复学号和异常金额记录。",
        ]
    )
