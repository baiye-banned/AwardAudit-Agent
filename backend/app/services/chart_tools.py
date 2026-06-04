from typing import Any
from urllib.parse import quote

import pandas as pd

from app.services.award_tools import infer_award_fields


MAX_CHARTS = 3


def build_charts(frame: pd.DataFrame, question: str = "") -> list[dict[str, Any]]:
    """根据问题生成最多 3 张“有业务含义”的图表。

    这里故意不再为了凑数量去画“前 10 行柱状图”。
    因为行号、序号、学号这类字段放到横轴上没有分析意义，
    面试时也很难解释清楚。
    """

    award_charts = _build_award_charts(frame, question)
    if award_charts:
        return award_charts[:MAX_CHARTS]

    generic_chart = _build_generic_group_chart(frame, question)
    if generic_chart:
        return [generic_chart]

    return []


def _build_award_charts(frame: pd.DataFrame, question: str) -> list[dict[str, Any]]:
    """为高校奖项名单生成审核场景下有意义的图表。"""

    fields = infer_award_fields(frame, use_llm=False)
    college_column = _field_column(fields, "college")
    department_column = _field_column(fields, "department")
    award_level_column = _field_column(fields, "award_level")
    amount_column = _field_column(fields, "amount")

    charts: list[dict[str, Any]] = []

    college_amount_chart = _group_sum_chart(
        frame=frame,
        group_column=college_column,
        value_column=amount_column,
        title="各书院总获奖额度 Top 10",
        x_label="书院",
        y_label="奖励金额合计",
    )

    if college_amount_chart:
        charts.append(college_amount_chart)

    # 如果用户问的是“哪个书院奖励总额最高”这一类具体问题，
    # 一张书院总额图就足够，继续补学院/等级图反而会分散注意力。
    if college_amount_chart and _is_specific_college_amount_question(question):
        return charts

    department_count_chart = _category_count_chart(
        frame=frame,
        column=department_column,
        title="各学院获奖人数 Top 10",
        x_label="学院",
        y_label="获奖记录数",
    )
    if department_count_chart:
        charts.append(department_count_chart)

    level_count_chart = _category_count_chart(
        frame=frame,
        column=award_level_column,
        title="奖项等级分布 Top 10",
        x_label="奖项等级",
        y_label="获奖记录数",
    )
    if level_count_chart:
        charts.append(level_count_chart)

    return charts


def _build_generic_group_chart(frame: pd.DataFrame, question: str) -> dict[str, Any] | None:
    """通用表格兜底：只画“分类列汇总数值列”的图。"""

    group_column = _find_meaningful_text_column(frame, question)
    value_column = _find_meaningful_numeric_column(frame, question)

    if not group_column or not value_column:
        return None

    return _group_sum_chart(
        frame=frame,
        group_column=group_column,
        value_column=value_column,
        title=f"按{group_column}汇总{value_column} Top 10",
        x_label=group_column,
        y_label=f"{value_column}合计",
    )


def _group_sum_chart(
    frame: pd.DataFrame,
    group_column: str | None,
    value_column: str | None,
    title: str,
    x_label: str,
    y_label: str,
) -> dict[str, Any] | None:
    if not group_column or not value_column:
        return None

    values = pd.to_numeric(frame[value_column], errors="coerce")
    temp = frame[[group_column]].copy()
    temp[value_column] = values
    temp = temp.dropna(subset=[group_column, value_column])

    if temp.empty:
        return None

    grouped = (
        temp.groupby(group_column)[value_column]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    if grouped.empty:
        return None

    svg = _bar_svg(
        title=title,
        labels=[str(label) for label in grouped.index],
        values=grouped.values.tolist(),
        x_label=x_label,
        y_label=y_label,
    )

    return {
        "title": title,
        "image_base64": _svg_data_url(svg),
        "insight": f"{grouped.index[0]} 的 {value_column} 最高，合计 {round(float(grouped.iloc[0]), 2)}。",
    }


def _category_count_chart(
    frame: pd.DataFrame,
    column: str | None,
    title: str,
    x_label: str,
    y_label: str,
) -> dict[str, Any] | None:
    if not column or _is_low_value_category(frame[column]):
        return None

    counts = frame[column].dropna().astype(str).value_counts().head(10)
    if counts.empty:
        return None

    svg = _bar_svg(
        title=title,
        labels=[str(label) for label in counts.index],
        values=counts.values.tolist(),
        x_label=x_label,
        y_label=y_label,
    )

    return {
        "title": title,
        "image_base64": _svg_data_url(svg),
        "insight": f"{counts.index[0]} 的记录数最多，共 {int(counts.iloc[0])} 条。",
    }


def _field_column(fields: dict[str, Any], key: str) -> str | None:
    field = fields.get(key, {})
    column = field.get("column")
    if isinstance(column, str) and column:
        return column
    return None


def _is_specific_college_amount_question(question: str) -> bool:
    return "书院" in question and any(
        word in question
        for word in ["总额", "总获奖", "奖励总额", "额度", "金额", "最高", "最多"]
    )


def _find_meaningful_text_column(frame: pd.DataFrame, question: str) -> str | None:
    text_columns = [
        str(column)
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]

    for column in text_columns:
        if column in question and _is_meaningful_category_column(frame, column):
            return column

    for column in text_columns:
        if _is_meaningful_category_column(frame, column):
            return column

    return None


def _find_meaningful_numeric_column(frame: pd.DataFrame, question: str) -> str | None:
    numeric_columns = [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]

    for column in numeric_columns:
        if column in question and _is_meaningful_numeric_column(column):
            return column

    for column in numeric_columns:
        if _is_meaningful_numeric_column(column):
            return column

    return None


def _is_meaningful_category_column(frame: pd.DataFrame, column: str) -> bool:
    if _looks_like_identifier_column(column):
        return False
    return not _is_low_value_category(frame[column])


def _is_meaningful_numeric_column(column: str) -> bool:
    return not _looks_like_identifier_column(column)


def _looks_like_identifier_column(column: str) -> bool:
    lowered = column.lower()
    keywords = ["序号", "学号", "编号", "id", "index", "姓名", "名称", "题目"]
    return any(keyword in lowered for keyword in keywords)


def _is_low_value_category(series: pd.Series) -> bool:
    cleaned = series.dropna().astype(str)
    if cleaned.empty:
        return True

    unique_count = cleaned.nunique()
    row_count = len(cleaned)

    if unique_count <= 1:
        return True

    # 唯一值太多时，通常是姓名、标题、编号，不适合做分布图。
    if row_count >= 10 and unique_count / row_count > 0.8:
        return True

    return False


def _bar_svg(
    title: str,
    labels: list[str],
    values: list[float],
    x_label: str,
    y_label: str,
) -> str:
    width = 760
    height = 420
    left = 80
    right = 720
    top = 78
    bottom = 300
    chart_height = bottom - top
    max_value = max([float(value) for value in values] + [1.0])
    slot_width = (right - left) / max(len(values), 1)
    bar_width = min(42, slot_width * 0.55)

    bars = []
    for index, raw_value in enumerate(values):
        value = float(raw_value)
        x = left + index * slot_width + (slot_width - bar_width) / 2
        bar_height = int(value / max_value * chart_height)
        y = bottom - bar_height
        label_x = x + bar_width / 2
        label = _short_text(labels[index])
        bars.append(
            f"<rect x='{x:.1f}' y='{y}' width='{bar_width:.1f}' height='{bar_height}' fill='#2563eb' rx='4' />"
            f"<text x='{label_x:.1f}' y='{bottom + 26}' text-anchor='end' font-size='11' "
            f"transform='rotate(-25 {label_x:.1f} {bottom + 26})'>{label}</text>"
        )

    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        "<rect width='100%' height='100%' fill='white' />"
        f"<text x='24' y='38' font-size='20' font-weight='700' fill='#111827'>{title}</text>"
        f"<text x='{left}' y='62' font-size='12' fill='#475569'>纵轴：{y_label}</text>"
        f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{bottom}' stroke='#d1d5db' />"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{bottom}' stroke='#d1d5db' />"
        f"{''.join(bars)}"
        f"<text x='{(left + right) / 2}' y='396' text-anchor='middle' font-size='12' fill='#475569'>横轴：{x_label}</text>"
        "</svg>"
    )


def _svg_data_url(svg: str) -> str:
    return f"data:image/svg+xml;charset=utf-8,{quote(svg)}"


def _short_text(text: str) -> str:
    if len(text) <= 8:
        return text
    return text[:8] + "..."
