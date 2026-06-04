import os
from typing import Any

import httpx
import pandas as pd
from langchain_core.tools import StructuredTool

from app.services.award_tools import (
    build_audit_report,
    infer_award_fields,
    quality_check_awards,
    query_college_awards,
    query_department_awards,
    query_person_awards,
    query_student_awards_by_id,
    rank_students_by_amount,
    search_award_records,
    summarize_awards,
)
from app.services.chart_tools import build_charts
from app.services.data_tools import (
    analyze_frame,
    build_data_profile,
    compare_groups,
    find_outliers,
)
from app.services.intent_router import choose_tools_with_llm, fallback_choose_tools
from app.services.sql_tools import answer_by_sql


def answer_question(
    question: str,
    frame: pd.DataFrame,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """智能报表分析 Agent 的主入口。

    当前版本用 LangChain 管理 tools，同时保留简单清晰的编排：
    1. 根据用户问题选择要调用的工具
    2. 用 LangChain StructuredTool 执行这些工具
    3. 把工具结果交给 LLM 或本地 fallback 总结

    这里没有让模型执行任意 pandas 代码，因为简历项目最重要的是稳定、可解释。
    """

    safe_history = history or []
    tools = create_analysis_tools(frame)
    selected_tools = choose_tools_with_llm(question, frame)
    tool_outputs = _run_tools(tools, selected_tools, question)

    answer = _summarize_with_llm_or_fallback(
        question=question,
        profile=tool_outputs["profile"],
        tool_results=tool_outputs["tool_results"],
        history=safe_history,
    )

    return {
        "answer": answer,
        "charts": tool_outputs["charts"],
        "profile": tool_outputs["profile"],
        "tool_trace": tool_outputs["tool_trace"],
        "audit_result": tool_outputs["audit_result"],
    }


def choose_tools(question: str) -> list[str]:
    """兼容旧测试和无 API Key 演示的兜底路由。

    主流程已经改为 choose_tools_with_llm(question, frame)。
    """

    return fallback_choose_tools(question)


def create_analysis_tools(frame: pd.DataFrame) -> list[StructuredTool]:
    """创建并注册 LangChain tools。"""

    def profile_data() -> dict[str, Any]:
        return build_data_profile(frame)

    def analyze_data() -> dict[str, Any]:
        return analyze_frame(frame)

    def chart_data(question: str = "") -> list[dict[str, Any]]:
        return build_charts(frame, question=question)

    def compare_group_data(question: str = "") -> dict[str, Any]:
        return compare_groups(frame, question=question)

    def find_outlier_data() -> dict[str, Any]:
        return find_outliers(frame)

    def infer_award_field_data() -> dict[str, Any]:
        return infer_award_fields(frame)

    def quality_check_award_data() -> dict[str, Any]:
        return quality_check_awards(frame)

    def award_summary_data() -> dict[str, Any]:
        return summarize_awards(frame)

    def audit_report_data() -> dict[str, Any]:
        return build_audit_report(frame)

    def query_person_award_data(question: str = "") -> dict[str, Any]:
        return query_person_awards(frame, question=question)

    def query_student_award_by_id_data(question: str = "") -> dict[str, Any]:
        return query_student_awards_by_id(frame, question=question)

    def query_college_award_data(question: str = "") -> dict[str, Any]:
        return query_college_awards(frame, question=question)

    def query_department_award_data(question: str = "") -> dict[str, Any]:
        return query_department_awards(frame, question=question)

    def rank_student_amount_data(question: str = "") -> dict[str, Any]:
        return rank_students_by_amount(frame, question=question)

    def search_award_record_data(question: str = "") -> dict[str, Any]:
        return search_award_records(frame, question=question)

    def answer_by_sql_data(question: str = "") -> dict[str, Any]:
        return answer_by_sql(frame, question=question)

    return [
        StructuredTool.from_function(
            func=profile_data,
            name="profile_data",
            description="读取表格的行数、列数、字段类型、样例行和数值列统计。",
        ),
        StructuredTool.from_function(
            func=analyze_data,
            name="analyze_data",
            description="使用 pandas 对表格做缺失值、数值列和分类列的基础分析。",
        ),
        StructuredTool.from_function(
            func=chart_data,
            name="build_charts",
            description="根据表格内容生成最多 3 张可视化图表。",
        ),
        StructuredTool.from_function(
            func=compare_group_data,
            name="compare_groups",
            description="按分类列分组，比较数值列的总和、均值和数量。",
        ),
        StructuredTool.from_function(
            func=find_outlier_data,
            name="find_outliers",
            description="使用 IQR 方法查找数值列里的异常值。",
        ),
        StructuredTool.from_function(
            func=infer_award_field_data,
            name="infer_award_fields",
            description="识别高校奖项名单中的学号、姓名、书院、学院、奖项等级、奖励金额等字段。",
        ),
        StructuredTool.from_function(
            func=quality_check_award_data,
            name="quality_check_awards",
            description="检查奖项名单中的缺失字段、重复学号、异常金额等审核问题。",
        ),
        StructuredTool.from_function(
            func=award_summary_data,
            name="award_summary",
            description="统计奖项名单的总金额、人均金额、书院/学院/奖项等级排行。",
        ),
        StructuredTool.from_function(
            func=audit_report_data,
            name="audit_report",
            description="生成高校奖项名单审核报告。",
        ),
        StructuredTool.from_function(
            func=query_person_award_data,
            name="query_person_awards",
            description="按姓名查询某个学生的获奖明细、记录数和总奖励金额。",
        ),
        StructuredTool.from_function(
            func=query_student_award_by_id_data,
            name="query_student_awards_by_id",
            description="按学号查询某个学生的获奖明细、记录数和总奖励金额。",
        ),
        StructuredTool.from_function(
            func=query_college_award_data,
            name="query_college_awards",
            description="查询某个书院的获奖人数、记录数、总奖励金额和等级分布。",
        ),
        StructuredTool.from_function(
            func=query_department_award_data,
            name="query_department_awards",
            description="查询某个学院的获奖人数、记录数、总奖励金额和等级分布。",
        ),
        StructuredTool.from_function(
            func=rank_student_amount_data,
            name="rank_students_by_amount",
            description="按学生汇总奖励金额并返回奖励金额最高的学生排行。",
        ),
        StructuredTool.from_function(
            func=search_award_record_data,
            name="search_award_records",
            description="按关键词搜索赛事名称、奖项等级、举办单位等文本字段中的获奖记录。",
        ),
        StructuredTool.from_function(
            func=answer_by_sql_data,
            name="answer_by_sql",
            description="把开放式统计问题转成安全只读 SQL，在临时 SQLite 表 awards 上查询。",
        ),
    ]


def _run_tools(tools: list[StructuredTool], selected_tools: list[str], question: str) -> dict[str, Any]:
    """执行被选中的 LangChain tools，并记录 tool_trace。"""

    tool_by_name = {tool.name: tool for tool in tools}
    tool_trace = []
    tool_results: dict[str, Any] = {}
    charts: list[dict[str, Any]] = []
    profile: dict[str, Any] | None = None
    audit_result: dict[str, Any] | None = None

    for tool_name in selected_tools:
        tool = tool_by_name[tool_name]
        if tool_name in [
            "build_charts",
            "compare_groups",
            "query_person_awards",
            "query_student_awards_by_id",
            "query_college_awards",
            "query_department_awards",
            "rank_students_by_amount",
            "search_award_records",
            "answer_by_sql",
        ]:
            result = tool.invoke({"question": question})
        else:
            result = tool.invoke({})
        tool_results[tool_name] = result
        tool_trace.append({"tool": tool_name, "status": "success"})

        if tool_name == "profile_data":
            profile = result
        elif tool_name == "build_charts":
            charts = result
        elif tool_name == "audit_report":
            audit_result = result

    # profile 是回答和前端展示的基础信息，所以即使路由没选中，也兜底补一次。
    if profile is None:
        profile = tool_by_name["profile_data"].invoke({})
        tool_results["profile_data"] = profile
        tool_trace.insert(0, {"tool": "profile_data", "status": "success"})

    return {
        "profile": profile,
        "charts": charts,
        "tool_results": tool_results,
        "tool_trace": tool_trace,
        "audit_result": audit_result,
    }


def _summarize_with_llm_or_fallback(
    question: str,
    profile: dict[str, Any],
    tool_results: dict[str, Any],
    history: list[dict[str, str]],
) -> str:
    """优先调用云端模型；未配置或调用失败时使用本地模板。"""

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return _fallback_answer(question, profile, tool_results)

    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    prompt = _build_prompt(question, profile, tool_results, history)

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个数据分析助手。请用中文回答，结论要简洁、具体、好懂。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return _fallback_answer(question, profile, tool_results)


def _build_prompt(
    question: str,
    profile: dict[str, Any],
    tool_results: dict[str, Any],
    history: list[dict[str, str]],
) -> str:
    recent_history = history[-4:]
    return (
        "用户正在分析一个表格数据集。\n"
        f"用户问题：{question}\n\n"
        f"数据概览：{profile}\n\n"
        f"LangChain tools 的执行结果：{tool_results}\n\n"
        f"最近对话：{recent_history}\n\n"
        "请输出：1）直接回答；2）关键发现；3）建议继续追问的问题。"
    )


def _fallback_answer(
    question: str,
    profile: dict[str, Any],
    tool_results: dict[str, Any],
) -> str:
    """没有 LLM 时的本地回答，保证项目没有 API Key 也能演示。"""

    direct_answer = _build_direct_answer(tool_results)
    lines = [
        f"回答：{direct_answer}" if direct_answer else "回答：我已经通过 LangChain tools 分析了这份表格。",
        f"- 数据规模：{profile['row_count']} 行，{profile['column_count']} 列。",
    ]

    if "analyze_data" in tool_results:
        missing_values = tool_results["analyze_data"]["missing_values"]
        if missing_values:
            lines.append(f"- 发现缺失值：{missing_values}。")
        else:
            lines.append("- 暂未发现明显缺失值。")

    if "compare_groups" in tool_results:
        group_result = tool_results["compare_groups"]
        groups = group_result.get("groups", [])
        if groups:
            top_group = groups[0]
            if not direct_answer:
                lines.append(
                    f"- {group_result['group_column']} 中，{top_group['group']} 的 "
                    f"{group_result['value_column']} 最高，总和为 {top_group['sum']}。"
                )
            top_text = "；".join(
                f"{item['group']}：{item['sum']}"
                for item in groups[:5]
            )
            lines.append(f"- Top 分组：{top_text}。")
        else:
            lines.append(f"- 分组对比结果：{group_result}。")

    if "find_outliers" in tool_results:
        lines.append(f"- 异常值检测结果：{tool_results['find_outliers']}。")

    if "build_charts" in tool_results:
        lines.append("- 已根据问题生成图表，可在页面下方查看。")

    if "infer_award_fields" in tool_results:
        fields = tool_results["infer_award_fields"]
        field_text = "；".join(
            f"{key}={value['column'] or '未识别'}({value['confidence']})"
            for key, value in fields.items()
        )
        lines.append(f"- 奖项字段识别：{field_text}。")

    if "quality_check_awards" in tool_results:
        quality = tool_results["quality_check_awards"]
        lines.append(f"- 审核问题数量：{quality['issue_count']}。")
        for issue in quality["quality_issues"][:5]:
            lines.append(f"  - {issue['message']}：{issue['count']} 条。")

    if "award_summary" in tool_results:
        summary = tool_results["award_summary"]
        lines.append(
            f"- 奖项汇总：共 {summary['total_rows']} 条记录，总奖励金额 {summary['total_amount']}，"
            f"人均 {summary['average_amount']}。"
        )

    if "audit_report" in tool_results:
        lines.append("- 已生成结构化审核报告，可在审核面板查看。")

    lines.append(f"- 你的问题是：{question}")
    lines.append("建议：可以继续追问“按类别比较”、“找异常值”或“生成趋势图”。")
    return "\n".join(lines)


def _build_direct_answer(tool_results: dict[str, Any]) -> str | None:
    """如果工具结果已经能直接回答问题，就把结论放到第一句。"""

    sql_result = tool_results.get("answer_by_sql")
    if sql_result and sql_result.get("success") and sql_result.get("rows"):
        return _format_sql_answer(sql_result["rows"][0])

    for tool_name in ["query_person_awards", "query_student_awards_by_id"]:
        result = tool_results.get(tool_name)
        if result and result.get("found"):
            return (
                f"{result['target']}一共获得 {result['total_amount']} 元奖励，"
                f"共 {result['record_count']} 条获奖记录。"
            )

    for tool_name in ["query_college_awards", "query_department_awards"]:
        result = tool_results.get(tool_name)
        if result and result.get("found"):
            return (
                f"{result['target']}共有 {result['record_count']} 条获奖记录，"
                f"{result.get('student_count', 0)} 名学生，奖励总额 {result['total_amount']} 元。"
            )

    ranking = tool_results.get("rank_students_by_amount")
    if ranking and ranking.get("rankings"):
        top_student = ranking["rankings"][0]
        name = top_student.get("name") or top_student.get("student_id") or "排名第一的学生"
        return (
            f"{name}的奖励总额最高，为 {top_student['total_amount']} 元，"
            f"共 {top_student['record_count']} 条记录。"
        )

    search = tool_results.get("search_award_records")
    if search and search.get("found"):
        return f"找到 {search['record_count']} 条包含“{search['keyword']}”的获奖记录。"

    group_result = tool_results.get("compare_groups")
    if not group_result:
        return None

    groups = group_result.get("groups", [])
    if not groups:
        return None

    top_group = groups[0]
    return (
        f"{top_group['group']}的{group_result['value_column']}最高，"
        f"总和为 {top_group['sum']}。"
    )


def _format_sql_answer(row: dict[str, Any]) -> str:
    if not row:
        return "SQL 查询没有返回结果。"

    first_value = next(iter(row.values()))
    details = "，".join(
        f"{key}={value}"
        for key, value in row.items()
        if value != first_value
    )

    if details:
        return f"SQL 查询结果显示：{first_value}，{details}。"
    return f"SQL 查询结果显示：{first_value}。"


def _should_route_person_query(text: str) -> bool:
    if "书院" in text or "学院" in text:
        return False

    person_words = ["一共", "总共", "多少钱", "拿了", "获得了", "这个人"]
    money_words = ["钱", "金额", "奖励", "额度"]
    return _contains_any(text, person_words) and _contains_any(text, money_words)


def _is_student_amount_ranking_question(text: str) -> bool:
    has_student_target = "谁" in text or "学生" in text or "人" in text
    has_ranking_word = "最多" in text or "最高" in text or "排名" in text or "top" in text
    has_amount_word = "钱" in text or "金额" in text or "奖励" in text or "额度" in text
    return has_student_target and has_ranking_word and has_amount_word


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _deduplicate(items: list[str]) -> list[str]:
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result
