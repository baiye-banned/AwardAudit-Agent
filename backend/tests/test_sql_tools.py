import pandas as pd
import pytest

from app.services.sql_tools import (
    answer_by_sql,
    build_sql_from_plan,
    generate_fallback_sql,
    parse_query_plan,
    run_select_sql,
    validate_select_sql,
)


def make_sql_frame():
    return pd.DataFrame(
        {
            "姓名": ["陈子玄", "陈子玄", "王赛博", "李四"],
            "学号": ["2024001", "2024001", "2024002", "2024003"],
            "住宿书院": ["思齐住宿书院", "思齐住宿书院", "知行住宿书院", "博艺住宿书院"],
            "奖励金额（元）": [300, 500, 600, 700],
        }
    )


def test_validate_select_sql_rejects_write_sql():
    with pytest.raises(ValueError):
        validate_select_sql("DROP TABLE awards")

    with pytest.raises(ValueError):
        validate_select_sql("SELECT * FROM awards; DELETE FROM awards")


def test_run_select_sql_executes_readonly_query():
    result = run_select_sql(
        make_sql_frame(),
        'SELECT "姓名", SUM("奖励金额（元）") AS total_amount FROM awards GROUP BY "姓名" ORDER BY total_amount DESC LIMIT 1',
    )

    assert result["rows"][0]["姓名"] == "陈子玄"
    assert result["rows"][0]["total_amount"] == 800


def test_generate_fallback_sql_handles_second_rank_question():
    sql = generate_fallback_sql(make_sql_frame(), "拿钱第二多的是谁")

    assert "LIMIT 1 OFFSET 1" in sql


def test_parse_query_plan_is_deterministic_for_rank_question():
    first = parse_query_plan(make_sql_frame(), "拿钱第二多的是谁")
    second = parse_query_plan(make_sql_frame(), "拿钱第二多的是谁")

    assert first == second
    assert first == {
        "intent": "rank_students",
        "entity": "student",
        "metric": "sum_amount",
        "rank_index": 2,
        "top_n": None,
    }


def test_build_sql_from_plan_is_deterministic():
    plan = {
        "intent": "rank_students",
        "entity": "student",
        "metric": "sum_amount",
        "rank_index": 2,
        "top_n": None,
    }

    first = build_sql_from_plan(make_sql_frame(), plan)
    second = build_sql_from_plan(make_sql_frame(), plan)

    assert first == second
    assert first.endswith('ORDER BY total_amount DESC, "姓名" ASC LIMIT 1 OFFSET 1')


def test_generate_fallback_sql_is_deterministic():
    first = generate_fallback_sql(make_sql_frame(), "拿钱第二多的是谁")
    second = generate_fallback_sql(make_sql_frame(), "拿钱第二多的是谁")

    assert first == second


def test_answer_by_sql_handles_award_count_ranking(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    frame = pd.DataFrame(
        {
            "姓名": ["陈子玄", "陈子玄", "王赛博", "李四"],
            "学号": ["2024001", "2024001", "2024002", "2024003"],
            "奖励金额（元）": [300, 500, 600, 700],
        }
    )

    result = answer_by_sql(frame, "获奖最多的学生是谁")

    assert result["success"] is True
    assert result["rows"][0]["姓名"] == "陈子玄"
    assert result["rows"][0]["award_count"] == 2


def test_answer_by_sql_handles_college_amount_ranking(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    frame = pd.DataFrame(
        {
            "姓名": ["陈子玄", "王赛博", "李四"],
            "住宿书院": ["思齐住宿书院", "知行住宿书院", "知行住宿书院"],
            "奖励金额（元）": [300, 600, 700],
        }
    )

    result = answer_by_sql(frame, "哪个书院奖励总额最高")

    assert result["success"] is True
    assert result["rows"][0]["住宿书院"] == "知行住宿书院"
    assert result["rows"][0]["total_amount"] == 1300


def test_answer_by_sql_answers_second_rank_question_without_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    result = answer_by_sql(make_sql_frame(), "拿钱第二多的是谁")

    assert result["success"] is True
    assert result["rows"][0]["姓名"] == "李四"
    assert result["rows"][0]["total_amount"] == 700
    assert "LIMIT 1 OFFSET 1" in result["sql"]
