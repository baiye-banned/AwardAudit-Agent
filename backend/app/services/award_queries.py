import json
import re
from typing import Any

import httpx
import pandas as pd

from app.services.award_common import (
    ACTION_KEYWORDS,
    ACTION_LABELS,
    FIELD_LABELS,
    LEVEL_GRADES,
    LEVEL_TIERS,
    _build_entity_award_result,
    _build_group_award_result,
    _clean_optional_text,
    _contains_any,
    _drop_rows_without_student_identity,
    _extract_existing_value,
    _field_column,
    _filter_by_text_value,
    _find_column_by_keywords,
    _not_found_result,
    _numeric_series,
    _records_from_frame,
    _round_number,
    _unsupported_award_answer,
)
from app.services.award_fields import infer_award_fields
from app.services.llm_config import get_llm_config


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
    query_plan = _build_award_query_plan(frame, question) if use_llm else None

    if query_plan:
        plan_result = _execute_award_query_plan(frame, fields, query_plan)
        if plan_result.get("answered"):
            return plan_result

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

def _build_award_query_plan(frame: pd.DataFrame, question: str) -> dict[str, Any] | None:
    llm_config = get_llm_config()
    if not llm_config.api_key:
        return None

    prompt = _build_query_plan_prompt(frame, question)

    try:
        response = httpx.post(
            f"{llm_config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {llm_config.api_key}"},
            json={
                "model": llm_config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你只返回 JSON，不要 Markdown，不要解释。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=20,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_query_plan_json(content)
    except Exception:
        return None

def _build_query_plan_prompt(frame: pd.DataFrame, question: str) -> str:
    profile = {
        "columns": [str(column) for column in frame.columns],
        "sample_rows": frame.head(3).where(pd.notna(frame), None).to_dict(orient="records"),
    }
    schema = {
        "intent": ["rank", "count", "unsupported"],
        "entity": ["student", "college", "department", "record"],
        "metric": ["sum_amount", "award_count", "record_count"],
        "rank_index": "positive integer, default 1",
    }

    return (
        "你是高校奖项名单分析 Agent 的查询计划生成器。\n"
        "请把用户问题分析成固定 JSON 查询计划，后端会用 pandas 执行，"
        "你不要计算金额，也不要输出 SQL 或代码。\n"
        f"允许的 JSON 字段和值：{json.dumps(schema, ensure_ascii=False)}\n"
        "如果用户问谁/哪位/学生/同学拿到奖金、补贴、资助额度、奖励金额最多，"
        "返回 intent=rank, entity=student, metric=sum_amount。\n"
        "如果用户问获奖最多/次数最多，返回 metric=award_count。\n"
        "如果无法归入这些固定意图，返回 intent=unsupported。\n"
        f"表格信息：{json.dumps(profile, ensure_ascii=False)}\n"
        f"用户问题：{question}\n"
        '示例：{"intent":"rank","entity":"student","metric":"sum_amount","rank_index":1}'
    )

def _parse_query_plan_json(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        text = text.removeprefix("json").strip()

    data = json.loads(text)
    if data.get("intent") not in ["rank", "count"]:
        return None

    plan = {
        "intent": data.get("intent"),
        "entity": data.get("entity"),
        "metric": data.get("metric"),
        "rank_index": data.get("rank_index", 1),
    }
    return _validate_query_plan(plan)

def _validate_query_plan(plan: dict[str, Any]) -> dict[str, Any] | None:
    if plan["intent"] == "rank":
        if plan["entity"] not in ["student", "college", "department"]:
            return None
        if plan["metric"] not in ["sum_amount", "award_count"]:
            return None
        try:
            rank_index = int(plan.get("rank_index") or 1)
        except (TypeError, ValueError):
            return None
        if rank_index < 1:
            return None
        plan["rank_index"] = rank_index
        return plan

    return None

def _execute_award_query_plan(
    frame: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    if plan["intent"] == "rank":
        return _rank_award_records_by_plan(frame, fields, plan)

    return _unsupported_award_answer(fields, "查询计划暂不支持该问题")

def _rank_award_records_by_plan(
    frame: pd.DataFrame,
    fields: dict[str, dict[str, Any]],
    plan: dict[str, Any],
) -> dict[str, Any]:
    return _rank_award_records(
        frame=frame,
        fields=fields,
        question="",
        entity=plan["entity"],
        metric=plan["metric"],
        rank_index=plan["rank_index"],
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
    entity: str | None = None,
    metric: str | None = None,
    rank_index: int | None = None,
) -> dict[str, Any]:
    entity = entity or _rank_entity(question)
    metric = metric or _rank_metric(question)
    rank_index = rank_index or _rank_index(question)

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
    return _contains_any(text, ["最多", "最高", "最大", "第一", "第二", "第三", "第", "top", "前"])

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
