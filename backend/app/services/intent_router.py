import json
from typing import Any

import httpx
import pandas as pd

from app.services.llm_config import get_llm_config


ALLOWED_TOOLS = [
    "profile_data",
    "analyze_data",
    "build_charts",
    "compare_groups",
    "find_outliers",
    "infer_award_fields",
    "quality_check_awards",
    "award_summary",
    "audit_report",
    "query_person_awards",
    "query_student_awards_by_id",
    "query_college_awards",
    "query_department_awards",
    "rank_students_by_amount",
    "search_award_records",
    "answer_award_question",
]


TOOL_DESCRIPTIONS = {
    "profile_data": "查看表格列名、行数、样例数据。",
    "analyze_data": "做缺失值、基础统计、整体数据质量概览。",
    "build_charts": "生成图表。",
    "compare_groups": "按书院、学院、地区、类别等分组汇总金额。",
    "find_outliers": "查找异常值。",
    "infer_award_fields": "识别奖项名单字段。",
    "quality_check_awards": "检查缺失、重复、异常金额等审核问题。",
    "award_summary": "生成奖项总览、总金额、人均金额、分布排行。",
    "audit_report": "生成审核报告。",
    "query_person_awards": "按姓名查询某个学生拿了多少钱、有哪些获奖记录。",
    "query_student_awards_by_id": "按学号查询某个学生拿了多少钱、有哪些获奖记录。",
    "query_college_awards": "查询某个书院的获奖情况。",
    "query_department_awards": "查询某个学院的获奖情况。",
    "rank_students_by_amount": "查询学生奖励金额排行。",
    "search_award_records": "按关键词搜索奖项记录。",
    "answer_award_question": "用 pandas 回答开放式统计、排行、次数、TopN、第几名和常见筛选问题。",
}


def choose_tools_with_llm(question: str, frame: pd.DataFrame) -> list[str]:
    """优先让 LLM 判断应该调用哪些 tools。

    LLM 只返回工具名 JSON；后端会过滤非法工具。
    没有 API Key 或调用失败时，使用 deterministic fallback。
    """

    llm_config = get_llm_config()
    if not llm_config.api_key:
        return fallback_choose_tools(question)

    prompt = _build_router_prompt(question, frame)

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
        tools = _parse_router_json(content)
        if tools:
            return _apply_pandas_query_hints(tools, question)
    except Exception:
        pass

    return _apply_pandas_query_hints(fallback_choose_tools(question), question)


def fallback_choose_tools(question: str) -> list[str]:
    """无 API Key 时的保守兜底。

    这里仍然是规则，但它只用于本地演示兜底，不再是主路由。
    """

    text = question.lower()
    selected = ["profile_data"]

    if _is_award_rank_or_count_question(text):
        selected.append("answer_award_question")
    elif _is_filtered_count_question(text):
        selected.append("answer_award_question")
    elif _is_person_amount_question(text):
        selected.append("query_person_awards")
    elif _is_student_id_amount_question(text):
        selected.append("query_student_awards_by_id")
    elif _is_group_compare_question(text):
        selected.append("compare_groups")
    elif any(word in text for word in ["图", "图表", "可视化", "chart", "plot"]):
        selected.extend(["analyze_data", "build_charts"])
    elif any(word in text for word in ["分析", "统计", "特点", "整体", "概览", "summary"]):
        selected.append("analyze_data")
    elif any(word in text for word in ["审核", "质检", "报告"]):
        selected.extend(["infer_award_fields", "quality_check_awards", "award_summary", "audit_report"])
    elif any(word in text for word in ["异常", "离群", "极端", "outlier"]):
        selected.extend(["analyze_data", "find_outliers"])
    elif any(word in text for word in ["找一下", "帮我找", "搜索", "包含", "相关"]):
        selected.append("search_award_records")

    return _deduplicate(selected)


def _build_router_prompt(question: str, frame: pd.DataFrame) -> str:
    profile = {
        "columns": [str(column) for column in frame.columns],
        "sample_rows": frame.head(3).where(pd.notna(frame), None).to_dict(orient="records"),
    }

    return (
        "你是一个高校奖项名单分析 Agent 的工具路由器。\n"
        "请根据用户问题选择需要调用的 tools。\n"
        "如果问题涉及开放式统计、排行、次数、TopN、第几名、复杂筛选，请选择 answer_award_question。\n"
        "如果问题是某个具体姓名/学号拿了多少钱，可以选择 query_person_awards 或 query_student_awards_by_id。\n"
        "如果问题是图表，请选择 build_charts。\n"
        "必须包含 profile_data。\n"
        f"可选 tools：{json.dumps(TOOL_DESCRIPTIONS, ensure_ascii=False)}\n"
        f"表格信息：{json.dumps(profile, ensure_ascii=False)}\n"
        f"用户问题：{question}\n"
        '只返回 JSON，例如 {"tools":["profile_data","answer_award_question"],"reason":"..."}'
    )


def _parse_router_json(content: str) -> list[str]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        text = text.removeprefix("json").strip()

    data = json.loads(text)
    raw_tools = data.get("tools", [])
    if not isinstance(raw_tools, list):
        return []

    filtered = [
        tool
        for tool in raw_tools
        if isinstance(tool, str) and tool in ALLOWED_TOOLS
    ]

    return _deduplicate(["profile_data", *filtered])


def _is_award_rank_or_count_question(text: str) -> bool:
    has_rank = any(word in text for word in ["最多", "最高", "第一", "第二", "第三", "第", "top", "前"])
    has_award = any(word in text for word in ["获奖", "奖励", "钱", "金额", "额度"])
    has_target = any(word in text for word in ["谁", "学生", "书院", "学院", "人"])
    return has_rank and has_award and has_target


def _is_person_amount_question(text: str) -> bool:
    has_amount = any(word in text for word in ["多少钱", "拿了", "一共", "总共", "奖励", "金额", "钱"])
    asks_ranking = any(word in text for word in ["谁", "最多", "最高", "第", "top", "前"])
    mentions_group = any(word in text for word in ["书院", "学院", "地区", "类别"])
    return has_amount and not asks_ranking and not mentions_group


def _is_group_compare_question(text: str) -> bool:
    has_group = any(word in text for word in ["按", "分组", "比较", "对比", "书院", "学院", "地区", "类别"])
    has_metric = any(word in text for word in ["销售额", "金额", "额度", "奖励", "获奖", "总额"])
    return has_group and has_metric


def _is_filtered_count_question(text: str) -> bool:
    has_count = any(word in text for word in ["有多少", "多少", "数量", "总数", "一共多少", "共有多少", "几个", "多少个", "多少条", "几条", "统计"])
    has_filter = any(word in text for word in ["申请", "申报", "提交", "审批", "审核", "公示", "公布", "通过", "被审批", "被申请", "被公示", "及以上", "以上", "以下", "级", "类"])
    return has_count and has_filter


def _is_student_id_amount_question(text: str) -> bool:
    has_student_id = "学号" in text or "编号" in text
    has_amount = any(word in text for word in ["多少钱", "金额", "奖励", "拿了", "总共", "一共"])
    return has_student_id and has_amount


def _apply_pandas_query_hints(selected_tools: list[str], question: str) -> list[str]:
    selected = _deduplicate(selected_tools)
    if "profile_data" not in selected:
        selected.insert(0, "profile_data")

    if (_is_award_rank_or_count_question(question) or _is_filtered_count_question(question)) and "answer_award_question" not in selected:
        selected.append("answer_award_question")

    return _deduplicate(selected)


def _deduplicate(items: list[str]) -> list[str]:
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
