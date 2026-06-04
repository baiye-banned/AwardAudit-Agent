import pandas as pd
from urllib.parse import unquote

from app.services.chart_tools import build_charts


def test_build_charts_groups_award_amount_by_college_question():
    frame = pd.DataFrame(
        {
            "序号": [1, 2, 3, 4],
            "住宿书院": ["思齐住宿书院", "知行住宿书院", "思齐住宿书院", "博艺住宿书院"],
            "奖励额度": [300, 500, 200, 100],
        }
    )

    charts = build_charts(frame, question="构建图表看看每个书院的总获奖额度")

    assert charts
    assert charts[0]["title"] == "各书院总获奖额度 Top 10"
    svg_text = unquote(charts[0]["image_base64"])
    assert "思齐住宿书院" in svg_text
    assert "知行住宿书院" in svg_text


def test_award_charts_do_not_include_row_index_or_order_number_charts():
    frame = pd.DataFrame(
        {
            "序号": [1, 2, 3, 4, 5],
            "学号": ["2024001", "2024002", "2024003", "2024004", "2024005"],
            "姓名": ["张三", "李四", "王五", "赵六", "钱七"],
            "住宿书院": ["思齐住宿书院", "知行住宿书院", "思齐住宿书院", "博艺住宿书院", "知行住宿书院"],
            "专业学院": ["体育学院", "经管学院", "体育学院", "新闻学院", "经管学院"],
            "获奖等级": ["一等奖", "二等奖", "二等奖", "三等奖", "一等奖"],
            "奖励金额（元）": [300, 500, 200, 100, 400],
        }
    )

    charts = build_charts(frame, question="请生成图表展示关键指标")
    titles = [chart["title"] for chart in charts]
    svg_text = "\n".join(unquote(chart["image_base64"]) for chart in charts)

    assert "各书院总获奖额度 Top 10" in titles
    assert all("序号" not in title for title in titles)
    assert all("前 10 行" not in title for title in titles)
    assert "横轴：书院" in svg_text
    assert "纵轴：奖励金额合计" in svg_text


def test_specific_college_amount_question_only_returns_relevant_chart():
    frame = pd.DataFrame(
        {
            "序号": [1, 2, 3, 4],
            "住宿书院": ["思齐住宿书院", "知行住宿书院", "思齐住宿书院", "博艺住宿书院"],
            "奖励金额（元）": [300, 500, 200, 100],
        }
    )

    charts = build_charts(frame, question="哪个书院奖励总额最高？请生成图表")

    assert [chart["title"] for chart in charts] == ["各书院总获奖额度 Top 10"]
