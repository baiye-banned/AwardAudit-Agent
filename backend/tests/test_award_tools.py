import pandas as pd

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


def make_award_frame():
    return pd.DataFrame(
        {
            "学生编号": ["2024001", "2024002", "2024002", ""],
            "学生姓名": ["张三", "李四", "李四", ""],
            "宿舍书院": ["思齐住宿书院", "知行住宿书院", "知行住宿书院", None],
            "专业学院": ["体育学院", "经济与管理学院", "经济与管理学院", "工学部"],
            "奖项级别": ["省级", "国家级", "国家级", "校级"],
            "奖励金额": [300, 500, 9000, None],
            "举办单位": ["河南省教育厅", "教育部", "教育部", "学校"],
        }
    )


def test_infer_award_fields_fallback_recognizes_flexible_names():
    result = infer_award_fields(make_award_frame(), use_llm=False)

    assert result["student_id"]["column"] == "学生编号"
    assert result["name"]["column"] == "学生姓名"
    assert result["college"]["column"] == "宿舍书院"
    assert result["department"]["column"] == "专业学院"
    assert result["award_level"]["column"] == "奖项级别"
    assert result["amount"]["column"] == "奖励金额"
    assert result["organizer"]["column"] == "举办单位"


def test_quality_check_awards_finds_missing_duplicate_and_amount_issues():
    result = quality_check_awards(make_award_frame(), use_llm=False)
    issue_types = [issue["type"] for issue in result["quality_issues"]]

    assert "missing_student_id" in issue_types
    assert "missing_name" in issue_types
    assert "missing_college" in issue_types
    assert "missing_amount" in issue_types
    assert "duplicate_student_id" in issue_types
    assert "amount_outlier" in issue_types


def test_summarize_awards_groups_amount_and_counts_correctly():
    result = summarize_awards(make_award_frame(), use_llm=False)

    assert result["total_rows"] == 4
    assert result["total_amount"] == 9800
    assert result["college_amount_top"][0]["group"] == "知行住宿书院"
    assert result["college_amount_top"][0]["sum"] == 9500
    assert result["department_count_top"][0]["group"] == "经济与管理学院"
    assert result["award_level_distribution"]["国家级"] == 2


def test_build_audit_report_returns_structured_report():
    result = build_audit_report(make_award_frame(), use_llm=False)

    assert result["fields"]["college"]["column"] == "宿舍书院"
    assert result["summary"]["total_amount"] == 9800
    assert result["quality_issues"]
    assert "审核报告" in result["report"]


def test_query_person_awards_sums_amount_and_returns_records():
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

    result = query_person_awards(frame, question="帮我看看王赛博一共拿了多少钱", use_llm=False)

    assert result["found"] is True
    assert result["target"] == "王赛博"
    assert result["record_count"] == 2
    assert result["total_amount"] == 800
    assert len(result["records"]) == 2


def test_query_student_awards_by_id_uses_student_id():
    result = query_student_awards_by_id(make_award_frame(), question="查一下 2024002 拿了多少钱", use_llm=False)

    assert result["found"] is True
    assert result["target"] == "2024002"
    assert result["record_count"] == 2
    assert result["total_amount"] == 9500


def test_query_college_and_department_awards():
    college_result = query_college_awards(make_award_frame(), question="知行住宿书院情况如何", use_llm=False)
    department_result = query_department_awards(make_award_frame(), question="经济与管理学院获奖情况", use_llm=False)

    assert college_result["found"] is True
    assert college_result["total_amount"] == 9500
    assert department_result["found"] is True
    assert department_result["record_count"] == 2


def test_rank_students_by_amount_returns_top_students():
    result = rank_students_by_amount(make_award_frame(), question="谁拿的钱最多", use_llm=False)

    assert result["rankings"][0]["student_id"] == "2024002"
    assert result["rankings"][0]["name"] == "李四"
    assert result["rankings"][0]["total_amount"] == 9500


def test_rank_students_by_amount_ignores_missing_student_identity():
    frame = pd.DataFrame(
        {
            "学号": [None, "2024001", "2024001"],
            "姓名": [None, "王赛博", "王赛博"],
            "住宿书院": ["思齐住宿书院", "思齐住宿书院", "思齐住宿书院"],
            "专业学院": ["体育学院", "体育学院", "体育学院"],
            "奖励金额（元）": [1707700, 300, 500],
        }
    )

    result = rank_students_by_amount(frame, question="拿钱最多的是谁,拿了多少", use_llm=False)

    assert result["rankings"][0]["name"] == "王赛博"
    assert result["rankings"][0]["total_amount"] == 800


def test_search_award_records_finds_keyword_matches():
    frame = pd.DataFrame(
        {
            "学号": ["2024001", "2024002"],
            "姓名": ["王赛博", "李四"],
            "住宿书院": ["思齐住宿书院", "知行住宿书院"],
            "专业学院": ["体育学院", "经管学院"],
            "获奖等级": ["一等奖", "二等奖"],
            "奖励金额（元）": [300, 500],
            "赛事名称": ["全国数学建模竞赛", "英语竞赛"],
        }
    )

    result = search_award_records(frame, question="找一下数学建模相关奖项", use_llm=False)

    assert result["found"] is True
    assert result["keyword"] == "数学建模"
    assert result["record_count"] == 1
    assert result["records"][0]["姓名"] == "王赛博"
