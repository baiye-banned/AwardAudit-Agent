from fastapi.testclient import TestClient

from app.main import app, run_server


client = TestClient(app)


def test_main_exposes_run_server_function():
    assert callable(run_server)


def test_main_file_can_be_executed_directly_for_help_check():
    from pathlib import Path

    main_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
    assert main_path.exists()


def test_upload_and_chat_smoke_flow():
    upload_response = client.post(
        "/api/upload",
        files={"file": ("sales.csv", b"month,sales\nJan,100\nFeb,180\n", "text/csv")},
    )

    assert upload_response.status_code == 200
    session_id = upload_response.json()["session_id"]

    chat_response = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "question": "这个数据有什么特点？",
            "history": [],
        },
    )

    assert chat_response.status_code == 200
    body = chat_response.json()
    assert body["answer"]
    assert "charts" in body
    assert [item["tool"] for item in body["tool_trace"]] == [
        "profile_data",
        "analyze_data",
    ]


def test_upload_award_file_returns_inferred_award_fields():
    upload_response = client.post(
        "/api/upload",
        files={
            "file": (
                "awards.csv",
                "学生编号,学生姓名,宿舍书院,专业学院,奖励金额\n"
                "2024001,张三,思齐住宿书院,体育学院,300\n".encode("utf-8"),
                "text/csv",
            )
        },
    )

    assert upload_response.status_code == 200
    fields = upload_response.json()["profile"]["award_fields"]
    assert fields["student_id"]["column"] == "学生编号"
    assert fields["college"]["column"] == "宿舍书院"
    assert fields["amount"]["column"] == "奖励金额"


def test_chat_audit_report_returns_audit_result():
    upload_response = client.post(
        "/api/upload",
        files={
            "file": (
                "awards.csv",
                "学生编号,学生姓名,宿舍书院,专业学院,奖励金额\n"
                "2024001,张三,思齐住宿书院,体育学院,300\n"
                "2024001,张三,思齐住宿书院,体育学院,300\n".encode("utf-8"),
                "text/csv",
            )
        },
    )
    session_id = upload_response.json()["session_id"]

    chat_response = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "question": "生成审核报告",
            "history": [],
        },
    )

    body = chat_response.json()
    assert chat_response.status_code == 200
    assert "audit_result" in body
    assert "audit_report" in [item["tool"] for item in body["tool_trace"]]
    assert body["audit_result"]["quality_issues"]


def test_chat_chart_question_runs_chart_tool():
    upload_response = client.post(
        "/api/upload",
        files={"file": ("sales.csv", b"month,sales,cost\nJan,100,60\nFeb,180,90\n", "text/csv")},
    )
    session_id = upload_response.json()["session_id"]

    chat_response = client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "question": "请分析销售额和成本，并生成图表",
            "history": [],
        },
    )

    body = chat_response.json()
    assert chat_response.status_code == 200
    assert [item["tool"] for item in body["tool_trace"]] == [
        "profile_data",
        "analyze_data",
        "build_charts",
    ]
    assert len(body["charts"]) >= 1
