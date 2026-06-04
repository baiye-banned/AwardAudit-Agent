import json

import httpx
import pandas as pd

from app.services.intent_router import choose_tools_with_llm, fallback_choose_tools


def test_fallback_routes_award_count_student_question_to_sql():
    selected = fallback_choose_tools("获奖最多的学生是谁")

    assert selected == ["profile_data", "answer_by_sql"]


def test_llm_router_uses_model_json_decision(monkeypatch):
    frame = pd.DataFrame(
        {
            "姓名": ["陈子玄", "王赛博"],
            "奖励金额（元）": [800, 600],
        }
    )

    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def fake_post(*args, **kwargs):
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tools": ["profile_data", "answer_by_sql"],
                                    "reason": "用户询问学生获奖次数排行，需要 SQL 聚合",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    selected = choose_tools_with_llm("获奖最多的学生是谁", frame)

    assert selected == ["profile_data", "answer_by_sql"]


def test_llm_router_rejects_unknown_tools(monkeypatch):
    frame = pd.DataFrame({"姓名": ["陈子玄"], "奖励金额（元）": [800]})
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    def fake_post(*args, **kwargs):
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": '{"tools":["drop_database","answer_by_sql"]}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    selected = choose_tools_with_llm("获奖最多的学生是谁", frame)

    assert selected == ["profile_data", "answer_by_sql"]
