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
        "answer_award_question",
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


def test_choose_tools_routes_college_award_question_to_pandas_query():
    selected = choose_tools("找出哪个书院的总获奖额度最多")

    assert selected == ["profile_data", "answer_award_question"]


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

    assert "answer_award_question" in [item["tool"] for item in result["tool_trace"]]
    assert result["answer"].splitlines()[0] == "回答：思齐住宿书院的奖励总额最高，为 500.0 元。"
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
    assert "answer_award_question" in choose_tools("谁拿的钱最多")
    assert "search_award_records" in choose_tools("找一下数学建模相关奖项")


def test_money_ranking_question_does_not_route_to_college_group_compare():
    selected = choose_tools("拿钱最多的是谁,拿了多少")

    assert "answer_award_question" in selected
    assert "compare_groups" not in selected


def test_second_rank_amount_question_routes_to_pandas_query_tool():
    selected = choose_tools("拿钱第二多的是谁")

    assert "answer_award_question" in selected


def test_answer_question_uses_pandas_for_second_rank_amount_question():
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

    assert "answer_award_question" in [item["tool"] for item in result["tool_trace"]]
    assert result["answer"].splitlines()[0] == "回答：李四的奖励总额第 2 高，为 700.0 元。"


def test_answer_question_uses_llm_intent_for_bonus_amount_ranking(monkeypatch):
    frame = pd.DataFrame(
        {
            "姓名": ["陈子玄", "王赛博", "李四"],
            "学号": ["2024001", "2024002", "2024003"],
            "奖励金额（元）": [300, 600, 700],
        }
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_post(*args, **kwargs):
        url = args[0]
        if "chat/completions" not in url:
            raise AssertionError("unexpected url")

        messages = kwargs["json"]["messages"]
        prompt = messages[-1]["content"]
        if "工具路由器" in prompt:
            content = '{"intent":"award_question","tools":["profile_data"]}'
        elif "查询计划" in prompt:
            content = '{"intent":"rank","entity":"student","metric":"sum_amount","rank_index":1}'
        else:
            raise RuntimeError("skip answer summary llm")

        request = httpx.Request("POST", url)
        return httpx.Response(
            status_code=200,
            request=request,
            json={"choices": [{"message": {"content": content}}]},
        )

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    result = answer_question(
        question="哪位同学获得资助额度最大",
        frame=frame,
        history=[],
    )

    assert "answer_award_question" in [item["tool"] for item in result["tool_trace"]]
    assert result["answer"].splitlines()[0] == "回答：李四的奖励总额最高，为 700.0 元。"


def test_answer_question_prefers_direct_pandas_answer_over_llm_summary(monkeypatch):
    frame = pd.DataFrame(
        {
            "姓名": ["陈子玄", "王赛博", "李四"],
            "学号": ["2024001", "2024002", "2024003"],
            "奖励金额（元）": [300, 600, 700],
        }
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_post(*args, **kwargs):
        url = args[0]
        messages = kwargs["json"]["messages"]
        prompt = messages[-1]["content"]
        if "工具路由器" in prompt:
            content = '{"intent":"award_question","tools":["profile_data"]}'
        elif "查询计划" in prompt:
            content = '{"intent":"rank","entity":"student","metric":"sum_amount","rank_index":1}'
        else:
            content = "错误总结：我已经分析了表格，但没有直接答案。"

        return httpx.Response(
            status_code=200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": content}}]},
        )

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    result = answer_question(
        question="哪位同学获得资助额度最大",
        frame=frame,
        history=[],
    )

    assert result["answer"].splitlines()[0] == "回答：李四的奖励总额最高，为 700.0 元。"


def test_llm_summary_prompt_does_not_include_all_profile_rows(monkeypatch):
    frame = pd.DataFrame(
        {
            "姓名": ["张三", "李四", "王五"],
            "学号": ["secret-1", "secret-2", "secret-3"],
            "奖励金额（元）": [300, 600, 700],
        }
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    prompts = []

    def fake_post(*args, **kwargs):
        url = args[0]
        prompt = kwargs["json"]["messages"][-1]["content"]
        prompts.append(prompt)

        if "工具路由器" in prompt:
            content = '{"tools":["profile_data","analyze_data"]}'
        else:
            content = "普通总结"

        return httpx.Response(
            status_code=200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": content}}]},
        )

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    answer_question("分析一下整体情况", frame=frame, history=[])

    summary_prompt = prompts[-1]
    assert "all_rows" not in summary_prompt
    assert "secret-3" not in summary_prompt


def test_answer_question_counts_filtered_approved_awards():
    frame = pd.DataFrame(
        {
            "姓名": ["张三", "李四", "王五", "赵六"],
            "获奖等级/收录情况": ["国A级", "省B级", "国家级A类", "国A级"],
            "奖励金额（元）": [300, 500, 200, 800],
            "审批状态": ["已审批", "已审批", "已审批", "待审批"],
        }
    )

    result = answer_question(
        question="帮我看看有几个国A级及以上的奖被审批了",
        frame=frame,
        history=[],
    )

    assert "answer_award_question" in [item["tool"] for item in result["tool_trace"]]
    assert result["answer"].splitlines()[0] == "回答：国A级及以上的奖在当前名单中共有 2 条。"


def test_answer_question_counts_national_b_or_above_awards():
    frame = pd.DataFrame(
        {
            "姓名": ["张三", "李四", "王五", "赵六"],
            "获奖等级/收录情况": ["国A级", "国B级", "国C级", "省A级"],
            "奖励金额（元）": [300, 500, 200, 800],
        }
    )

    result = answer_question(
        question="有多少国b级及以上的奖被申请了",
        frame=frame,
        history=[],
    )

    assert "answer_award_question" in [item["tool"] for item in result["tool_trace"]]
    assert result["answer"].splitlines()[0] == "回答：国B级及以上的奖在当前名单中共有 2 条。"


def test_answer_question_counts_exact_national_a_awards():
    frame = pd.DataFrame(
        {
            "姓名": ["张三", "李四", "王五", "赵六"],
            "获奖等级/收录情况": ["一等奖", "二等奖", "三等奖", "一等奖"],
            "审批依据": ["国A", "国B", "国A+", "国A（团队10人，2倍奖励）"],
        }
    )

    result = answer_question(
        question="国a级有几条",
        frame=frame,
        history=[],
    )

    assert "answer_award_question" in [item["tool"] for item in result["tool_trace"]]
    assert result["answer"].splitlines()[0] == "回答：国A级的奖在当前名单中共有 2 条。"


def test_same_pandas_question_returns_same_pandas_and_answer():
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
