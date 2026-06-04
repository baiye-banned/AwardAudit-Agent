import pandas as pd

from app.services.agent import answer_question, choose_tools, create_analysis_tools


def test_answer_question_returns_text_and_charts_for_numeric_data():
    frame = pd.DataFrame(
        {
            "month": ["Jan", "Feb", "Mar"],
            "sales": [100, 180, 140],
            "cost": [60, 90, 80],
        }
    )

    result = answer_question(
        question="请分析销售额和成本，并生成图表",
        frame=frame,
        history=[],
    )

    assert "回答" in result["answer"]
    assert len(result["charts"]) >= 1
    assert len(result["charts"]) <= 3
    assert result["charts"][0]["image_base64"].startswith("data:image/svg+xml")


def test_agent_registers_langchain_tools():
    frame = pd.DataFrame(
        {
            "month": ["Jan", "Feb"],
            "sales": [100, 180],
        }
    )

    tools = create_analysis_tools(frame)
    tool_names = [tool.name for tool in tools]

    assert tool_names == [
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
        "answer_by_sql",
    ]
    assert all(hasattr(tool, "invoke") for tool in tools)


def test_choose_tools_routes_profile_question_to_profile_only():
    selected = choose_tools("这个表有哪些列？")

    assert selected == ["profile_data"]


def test_choose_tools_routes_chart_question_to_charts():
    selected = choose_tools("请分析销售额和成本，并生成图表")

    assert selected == ["profile_data", "analyze_data", "build_charts"]


def test_choose_tools_routes_business_question_to_group_compare():
    selected = choose_tools("请按地区比较销售额")

    assert "compare_groups" in selected


def test_choose_tools_routes_outlier_question_to_outlier_tool():
    selected = choose_tools("帮我找一下异常值")

    assert "find_outliers" in selected


def test_answer_question_only_runs_selected_tools():
    frame = pd.DataFrame(
        {
            "region": ["North", "South"],
            "sales": [100, 180],
        }
    )

    result = answer_question(
        question="这个表有哪些列？",
        frame=frame,
        history=[],
    )

    assert [item["tool"] for item in result["tool_trace"]] == ["profile_data"]
    assert result["charts"] == []


def test_choose_tools_routes_college_award_question_to_sql():
    selected = choose_tools("找出哪个书院的总获奖额度最多")

    assert selected == ["profile_data", "answer_by_sql"]


def test_answer_question_compares_award_amount_by_college():
    frame = pd.DataFrame(
        {
            "序号": [1, 2, 3, 4],
            "住宿书院": ["思齐住宿书院", "知行住宿书院", "思齐住宿书院", "博艺住宿书院"],
            "奖励额度": [300, 500, 200, 100],
        }
    )

    result = answer_question(
        question="找出哪个书院的总获奖额度最多",
        frame=frame,
        history=[],
    )

    assert "answer_by_sql" in [item["tool"] for item in result["tool_trace"]]
    assert result["answer"].splitlines()[0] == "回答：SQL 查询结果显示：思齐住宿书院，total_amount=500，record_count=2。"
    assert "思齐住宿书院" in result["answer"]
    assert "发现缺失值" not in result["answer"]


def test_answer_question_runs_audit_report_tool():
    frame = pd.DataFrame(
        {
            "学生编号": ["2024001", "2024001"],
            "学生姓名": ["张三", "张三"],
            "宿舍书院": ["思齐住宿书院", "思齐住宿书院"],
            "专业学院": ["体育学院", "体育学院"],
            "奖励金额": [300, 300],
        }
    )

    result = answer_question(
        question="请生成审核报告",
        frame=frame,
        history=[],
    )

    assert "audit_report" in [item["tool"] for item in result["tool_trace"]]
    assert result["audit_result"]["quality_issues"]
    assert "审核报告" in result["audit_result"]["report"]


def test_choose_tools_routes_person_amount_question():
    selected = choose_tools("帮我看看王赛博一共拿了多少钱")

    assert "query_person_awards" in selected


def test_answer_question_sums_person_awards():
    frame = pd.DataFrame(
        {
            "学号": ["2024001", "2024001", "2024002"],
            "姓名": ["王赛博", "王赛博", "李四"],
            "住宿书院": ["思齐住宿书院", "思齐住宿书院", "知行住宿书院"],
            "专业学院": ["体育学院", "体育学院", "经管学院"],
            "获奖等级": ["一等奖", "二等奖", "三等奖"],
            "奖励金额（元）": [300, 500, 200],
            "赛事名称": ["数学建模", "程序设计", "英语竞赛"],
        }
    )

    result = answer_question(
        question="帮我看看王赛博一共拿了多少钱",
        frame=frame,
        history=[],
    )

    assert "query_person_awards" in [item["tool"] for item in result["tool_trace"]]
    assert result["answer"].splitlines()[0] == "回答：王赛博一共获得 800.0 元奖励，共 2 条获奖记录。"


def test_choose_tools_routes_student_ranking_and_keyword_search():
    assert "answer_by_sql" in choose_tools("谁拿的钱最多")
    assert "search_award_records" in choose_tools("找一下数学建模相关奖项")


def test_money_ranking_question_does_not_route_to_college_group_compare():
    selected = choose_tools("拿钱最多的是谁,拿了多少")

    assert "answer_by_sql" in selected
    assert "compare_groups" not in selected


def test_second_rank_amount_question_routes_to_sql_tool():
    selected = choose_tools("拿钱第二多的是谁")

    assert "answer_by_sql" in selected


def test_answer_question_uses_sql_for_second_rank_amount_question():
    frame = pd.DataFrame(
        {
            "姓名": ["陈子玄", "陈子玄", "王赛博", "李四"],
            "学号": ["2024001", "2024001", "2024002", "2024003"],
            "住宿书院": ["思齐住宿书院", "思齐住宿书院", "知行住宿书院", "博艺住宿书院"],
            "奖励金额（元）": [300, 500, 600, 700],
        }
    )

    result = answer_question(
        question="拿钱第二多的是谁",
        frame=frame,
        history=[],
    )

    assert "answer_by_sql" in [item["tool"] for item in result["tool_trace"]]
    assert result["answer"].splitlines()[0] == "回答：SQL 查询结果显示：李四，total_amount=700，record_count=1。"


def test_same_sql_question_returns_same_sql_and_answer():
    frame = pd.DataFrame(
        {
            "姓名": ["陈子玄", "陈子玄", "王赛博", "李四"],
            "学号": ["2024001", "2024001", "2024002", "2024003"],
            "住宿书院": ["思齐住宿书院", "思齐住宿书院", "知行住宿书院", "博艺住宿书院"],
            "奖励金额（元）": [300, 500, 600, 700],
        }
    )

    first = answer_question("拿钱第二多的是谁", frame=frame, history=[])
    second = answer_question("拿钱第二多的是谁", frame=frame, history=[])

    assert first["tool_trace"] == second["tool_trace"]
    assert first["answer"] == second["answer"]
