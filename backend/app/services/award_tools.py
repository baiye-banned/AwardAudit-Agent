import json
import re
from typing import Any

import httpx
import pandas as pd

from app.services.llm_config import get_llm_config


FIELD_LABELS = {
    "student_id": "学号",
    "name": "姓名",
    "college": "书院",
    "department": "学院",
    "award_level": "奖项等级",
    "amount": "奖励金额",
    "organizer": "举办单位",
}

FIELD_KEYWORDS = {
    "student_id": ["学号", "学生编号", "学生号", "student id", "id"],
    "name": ["姓名", "学生姓名", "名字", "name"],
    "college": ["住宿书院", "宿舍书院", "所在书院", "归属书院", "书院"],
    "department": ["专业学院", "所在学院", "学院", "院系", "学部", "department"],
    "award_level": ["获奖等级", "获奖级别", "奖项等级", "奖项级别", "收录情况", "级别"],
    "amount": ["奖励额度", "奖励金额", "获奖额度", "获奖金额", "金额", "额度"],
    "organizer": ["举办单位", "主办单位", "期刊名", "组织单位", "单位"],
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


def query_person_awards(frame: pd.DataFrame, question: str = "", use_llm: bool = True) -> dict[str, Any]:
    """查询某个学生的获奖明细和总奖励金额。

    例子：用户问“王赛博一共拿了多少钱”。
    这个工具会先找到姓名列，再从问题里找出真实出现过的姓名。
    """

    fields = infer_award_fields(frame, use_llm=use_llm)
    name_column = _field_column(fields, "name")
    amount_column = _field_column(fields, "amount")

    if not name_column:
        return _not_found_result("未识别到姓名字段")

    target = _extract_existing_value(frame, name_column, question)
    if not target:
        return _not_found_result("没有从问题中识别到学生姓名")

    matched = _filter_by_text_value(frame, name_column, target)
    return _build_entity_award_result(matched, fields, target, "person", amount_column)


def query_student_awards_by_id(frame: pd.DataFrame, question: str = "", use_llm: bool = True) -> dict[str, Any]:
    """按学号查询某个学生的获奖明细和总奖励金额。"""

    fields = infer_award_fields(frame, use_llm=use_llm)
    student_id_column = _field_column(fields, "student_id")
    amount_column = _field_column(fields, "amount")

    if not student_id_column:
        return _not_found_result("未识别到学号字段")

    target = _extract_existing_value(frame, student_id_column, question)
    if not target:
        return _not_found_result("没有从问题中识别到学号")

    matched = _filter_by_text_value(frame, student_id_column, target)
    return _build_entity_award_result(matched, fields, target, "student_id", amount_column)


def query_college_awards(frame: pd.DataFrame, question: str = "", use_llm: bool = True) -> dict[str, Any]:
    """查询某个住宿书院的获奖统计。"""

    fields = infer_award_fields(frame, use_llm=use_llm)
    college_column = _field_column(fields, "college")
    amount_column = _field_column(fields, "amount")

    if not college_column:
        return _not_found_result("未识别到书院字段")

    target = _extract_existing_value(frame, college_column, question)
    if not target:
        return _not_found_result("没有从问题中识别到书院名称")

    matched = _filter_by_text_value(frame, college_column, target)
    return _build_group_award_result(matched, fields, target, "college", amount_column)


def query_department_awards(frame: pd.DataFrame, question: str = "", use_llm: bool = True) -> dict[str, Any]:
    """查询某个专业学院的获奖统计。"""

    fields = infer_award_fields(frame, use_llm=use_llm)
    department_column = _field_column(fields, "department")
    amount_column = _field_column(fields, "amount")

    if not department_column:
        return _not_found_result("未识别到学院字段")

    target = _extract_existing_value(frame, department_column, question)
    if not target:
        return _not_found_result("没有从问题中识别到学院名称")

    matched = _filter_by_text_value(frame, department_column, target)
    return _build_group_award_result(matched, fields, target, "department", amount_column)


def rank_students_by_amount(frame: pd.DataFrame, question: str = "", use_llm: bool = True) -> dict[str, Any]:
    """按学生汇总奖励金额并排序。

    例子：用户问“谁拿的钱最多”。
    """

    fields = infer_award_fields(frame, use_llm=use_llm)
    student_id_column = _field_column(fields, "student_id")
    name_column = _field_column(fields, "name")
    amount_column = _field_column(fields, "amount")

    if not amount_column or not (student_id_column or name_column):
        return {"rankings": [], "message": "缺少学号/姓名或奖励金额字段，无法做学生金额排行"}

    group_columns = [column for column in [student_id_column, name_column] if column]
    temp = frame[group_columns].copy()
    temp[amount_column] = _numeric_series(frame, amount_column)
    temp = temp.dropna(subset=[amount_column])
    temp = _drop_rows_without_student_identity(temp, group_columns)

    if temp.empty:
        return {"rankings": [], "message": "没有同时具备学生身份和奖励金额的有效记录"}

    grouped = (
        temp.groupby(group_columns)[amount_column]
        .agg(["sum", "count"])
        .sort_values("sum", ascending=False)
        .head(10)
        .reset_index()
    )

    rankings = []
    for _, row in grouped.iterrows():
        rankings.append(
            {
                "student_id": _clean_optional_text(row[student_id_column]) if student_id_column else None,
                "name": _clean_optional_text(row[name_column]) if name_column else None,
                "total_amount": _round_number(row["sum"]),
                "record_count": int(row["count"]),
            }
        )

    return {
        "fields": fields,
        "rankings": rankings,
    }


def search_award_records(frame: pd.DataFrame, question: str = "", use_llm: bool = True) -> dict[str, Any]:
    """按关键词搜索奖项记录。

    例子：用户问“找一下数学建模相关奖项”。
    """

    fields = infer_award_fields(frame, use_llm=use_llm)
    keyword = _extract_search_keyword(question)
    if not keyword:
        return _not_found_result("没有从问题中识别到搜索关键词")

    text_columns = [
        column
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]

    mask = pd.Series(False, index=frame.index)
    for column in text_columns:
        values = frame[column].fillna("").astype(str)
        mask = mask | values.str.contains(keyword, case=False, regex=False)

    matched = frame[mask]
    return {
        "found": not matched.empty,
        "keyword": keyword,
        "record_count": int(len(matched)),
        "fields": fields,
        "records": _records_from_frame(matched.head(10)),
        "message": "" if not matched.empty else f"没有找到包含“{keyword}”的记录",
    }


def answer_award_question(frame: pd.DataFrame, question: str = "", use_llm: bool = True) -> dict[str, Any]:
    """用 pandas 回答奖项名单里的常见统计问题。

    这个函数替代旧 SQL 模块。项目当前只分析单个 CSV/XLSX，数据已经在
    DataFrame 里，所以直接用 pandas 计算更轻、更好讲，也更容易调试。
    """

    fields = infer_award_fields(frame, use_llm=use_llm)
    text = question.lower()

    if _looks_like_filtered_count_question(text):
        return _count_filtered_awards(frame, fields, question)

    if _looks_like_ranking_question(text):
        return _rank_award_records(frame, fields, question)

    return {
        "answered": False,
        "query_type": "unsupported",
        "message": "这个问题暂时不属于内置奖项统计工具的覆盖范围",
        "fields": fields,
    }


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


def _count_filtered_awards(
    frame: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    question: str,
) -> dict[str, Any]:
    level_threshold = _extract_level_threshold(question)
    level_column = _best_level_column(frame, fields, level_threshold)
    action = _extract_action_condition(question)
    matched = frame.copy()
    filters = []
    notes = []

    if level_threshold:
        if not level_column:
            return _unsupported_award_answer(fields, "缺少奖项等级字段，无法按等级条件计数")

        if level_threshold["mode"] == "at_least":
            mask = frame[level_column].apply(lambda value: _level_meets_threshold(value, level_threshold))
        else:
            mask = frame[level_column].apply(lambda value: _level_equals_threshold(value, level_threshold))
        matched = matched[mask]
        filters.append({"column": level_column, "condition": _level_condition_text(level_threshold)})

    if action:
        status_column = _find_status_column(frame, action)
        if status_column:
            status_values = matched[status_column].fillna("").astype(str)
            status_mask = _status_mask(status_values, action)
            matched = matched[status_mask]
            filters.append({"column": status_column, "condition": ACTION_LABELS[action]})
        else:
            notes.append("未发现状态列，按当前名单记录计数")

    condition_text = _build_condition_text(level_threshold)

    return {
        "answered": True,
        "query_type": "count",
        "record_count": int(len(matched)),
        "filters": filters,
        "condition_text": condition_text,
        "notes": notes,
        "fields": fields,
        "rows": [{"record_count": int(len(matched))}],
        "message": "",
    }


def _rank_award_records(
    frame: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    question: str,
) -> dict[str, Any]:
    entity = _rank_entity(question)
    metric = _rank_metric(question)
    rank_index = _rank_index(question)

    if entity == "college":
        group_columns = [_field_column(fields, "college")]
    elif entity == "department":
        group_columns = [_field_column(fields, "department")]
    else:
        student_id_column = _field_column(fields, "student_id")
        name_column = _field_column(fields, "name")
        group_columns = [column for column in [student_id_column, name_column] if column]

    group_columns = [column for column in group_columns if column]
    if not group_columns:
        return _unsupported_award_answer(fields, "缺少可用于排行的业务字段")

    temp = frame[group_columns].copy()
    temp = _drop_blank_group_rows(temp, group_columns)
    if temp.empty:
        return _unsupported_award_answer(fields, "没有可用于排行的有效记录")

    amount_column = _field_column(fields, "amount")
    if metric == "sum_amount":
        if not amount_column:
            return _unsupported_award_answer(fields, "缺少奖励金额字段，无法按金额排行")
        temp["__metric"] = _numeric_series(frame, amount_column)
        temp = temp.dropna(subset=["__metric"])
        grouped = temp.groupby(group_columns, dropna=True)["__metric"].agg(["sum", "count"])
        sort_column = "sum"
    else:
        grouped = temp.groupby(group_columns, dropna=True).size().to_frame("count")
        sort_column = "count"

    grouped = grouped.sort_values([sort_column], ascending=False).reset_index()
    if rank_index > len(grouped):
        return _unsupported_award_answer(fields, f"只有 {len(grouped)} 个可排行对象，无法返回第 {rank_index} 名")

    rows = []
    for _, row in grouped.head(10).iterrows():
        item = _build_ranking_row(row, group_columns, entity, metric)
        rows.append(item)

    selected_row = rows[rank_index - 1]
    return {
        "answered": True,
        "query_type": "rank",
        "entity": entity,
        "metric": metric,
        "rank_index": rank_index,
        "fields": fields,
        "rows": [selected_row],
        "rankings": rows,
        "message": "",
    }


def _build_ranking_row(
    row: pd.Series,
    group_columns: list[str],
    entity: str,
    metric: str,
) -> dict[str, Any]:
    item: dict[str, Any] = {"record_count": int(row["count"])}

    if entity == "student":
        if len(group_columns) >= 1:
            item["student_id"] = _clean_optional_text(row[group_columns[0]])
        if len(group_columns) >= 2:
            item["name"] = _clean_optional_text(row[group_columns[1]])
    else:
        item["group"] = _clean_optional_text(row[group_columns[0]])

    if metric == "sum_amount":
        item["total_amount"] = _round_number(row["sum"])
    else:
        item["award_count"] = int(row["count"])

    return item


def _looks_like_filtered_count_question(text: str) -> bool:
    return _contains_any(text, ["有多少", "多少", "数量", "总数", "一共多少", "共有多少", "几个", "多少个", "多少条", "几条"]) and _contains_any(
        text,
        ["申请", "申报", "提交", "审批", "审核", "公示", "公布", "通过", "及以上", "以上", "以下", "级", "类"],
    )


def _looks_like_ranking_question(text: str) -> bool:
    return _contains_any(text, ["最多", "最高", "第一", "第二", "第三", "第", "top", "前"])


def _rank_entity(question: str) -> str:
    if "书院" in question:
        return "college"
    if "学院" in question:
        return "department"
    return "student"


def _rank_metric(question: str) -> str:
    if _contains_any(question, ["几次", "次数", "获奖最多", "记录最多"]):
        return "award_count"
    return "sum_amount"


def _rank_index(question: str) -> int:
    if _contains_any(question, ["第二", "第2", "第 2"]):
        return 2
    if _contains_any(question, ["第三", "第3", "第 3"]):
        return 3
    return 1


def _extract_level_threshold(question: str) -> dict[str, Any] | None:
    text = _normalize_level_text(question)
    match = re.search(r"(国家|国|省|市|校)([ABC])", text)
    if match:
        tier_name = "国家" if match.group(1) in ["国家", "国"] else match.group(1)
        grade = match.group(2)
        return {
            "tier_name": tier_name,
            "display_tier": match.group(1),
            "tier_rank": LEVEL_TIERS[tier_name],
            "grade": grade,
            "grade_rank": LEVEL_GRADES[grade],
            "mode": "at_least" if _contains_any(question, ["及以上", "以上"]) else "exact",
        }
    return None


def _best_level_column(
    frame: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    threshold: dict[str, Any] | None,
) -> str | None:
    fallback_column = _field_column(fields, "award_level")
    if not threshold:
        return fallback_column

    best_column = None
    best_score = 0
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].dropna().astype(str).head(200)
        score = int(values.apply(_parse_award_level).notna().sum())
        if score > best_score:
            best_column = str(column)
            best_score = score

    if best_score > 0:
        return best_column
    return fallback_column


def _level_meets_threshold(value: Any, threshold: dict[str, Any]) -> bool:
    parsed = _parse_award_level(value)
    if not parsed:
        return False

    if parsed["tier_rank"] > threshold["tier_rank"]:
        return True
    if parsed["tier_rank"] < threshold["tier_rank"]:
        return False
    return parsed["grade_rank"] >= threshold["grade_rank"]


def _level_equals_threshold(value: Any, threshold: dict[str, Any]) -> bool:
    parsed = _parse_award_level(value)
    if not parsed:
        return False
    if "+" in str(value):
        return False
    return (
        parsed["tier_rank"] == threshold["tier_rank"]
        and parsed["grade_rank"] == threshold["grade_rank"]
    )


def _parse_award_level(value: Any) -> dict[str, Any] | None:
    text = _normalize_level_text(value)
    match = re.search(r"(国家|国|省|市|校)([ABC])", text)
    if match:
        tier_name = "国家" if match.group(1) in ["国家", "国"] else match.group(1)
        grade = match.group(2)
        return {
            "tier_name": tier_name,
            "tier_rank": LEVEL_TIERS[tier_name],
            "grade": grade,
            "grade_rank": LEVEL_GRADES[grade],
        }
    return None


def _normalize_level_text(value: Any) -> str:
    text = str(value).upper()
    for old, new in {
        "Ａ": "A",
        "Ｂ": "B",
        "Ｃ": "C",
        "国家级": "国家",
        "国家类": "国家",
        "省级": "省",
        "市级": "市",
        "校级": "校",
        "类": "",
        "级": "",
        "奖": "",
        " ": "",
        "\u3000": "",
    }.items():
        text = text.replace(old, new)
    return text


def _extract_action_condition(question: str) -> str | None:
    for action, keywords in ACTION_KEYWORDS.items():
        if _contains_any(question, keywords):
            return action
    return None


def _find_status_column(frame: pd.DataFrame, action: str) -> str | None:
    keywords = [*ACTION_KEYWORDS[action], "状态", "进度"]
    return _find_column_by_keywords(frame, keywords)


def _status_mask(values: pd.Series, action: str) -> pd.Series:
    allowed_words = ACTION_KEYWORDS[action]
    mask = pd.Series(False, index=values.index)
    for word in allowed_words:
        mask = mask | values.str.contains(word, regex=False)

    negative_words = ["未", "待", "不", "驳回", "退回", "失败"]
    for word in negative_words:
        mask = mask & ~values.str.contains(word, regex=False)

    return mask


def _level_condition_text(threshold: dict[str, Any]) -> str:
    display_tier = threshold.get("display_tier") or threshold["tier_name"]
    suffix = "级及以上" if threshold.get("mode") == "at_least" else "级"
    return f"{display_tier}{threshold['grade']}{suffix}"


def _build_condition_text(threshold: dict[str, Any] | None) -> str:
    if threshold:
        return f"{_level_condition_text(threshold)}的奖"
    return "符合条件的奖"


def _drop_blank_group_rows(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    keep_mask = pd.Series(True, index=frame.index)
    for column in group_columns:
        values = frame[column].apply(_clean_optional_text)
        keep_mask = keep_mask & values.notna()
    return frame[keep_mask]


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


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


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


def _records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _to_plain_value(value) for key, value in row.items()}
        for row in frame.where(pd.notna(frame), None).to_dict(orient="records")
    ]


def _drop_rows_without_student_identity(frame: pd.DataFrame, identity_columns: list[str]) -> pd.DataFrame:
    if not identity_columns:
        return frame

    keep_mask = pd.Series(False, index=frame.index)
    for column in identity_columns:
        values = frame[column].apply(_clean_optional_text)
        keep_mask = keep_mask | values.notna()

    return frame[keep_mask]


def _clean_optional_text(value: Any) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() in ["nan", "none", "null"]:
        return None

    return text


def _extract_search_keyword(question: str) -> str | None:
    text = question.strip()
    stop_words = [
        "找一下",
        "帮我找",
        "查询",
        "查一下",
        "相关奖项",
        "相关记录",
        "奖项",
        "记录",
        "有哪些",
        "关于",
        "的",
        "一下",
        "请",
    ]
    for word in stop_words:
        text = text.replace(word, "")

    text = text.strip(" ，。？！?：:")
    return text or None


def _not_found_result(message: str) -> dict[str, Any]:
    return {
        "found": False,
        "message": message,
        "record_count": 0,
        "total_amount": 0,
        "records": [],
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
