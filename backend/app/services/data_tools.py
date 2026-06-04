from io import BytesIO
from typing import Any

import pandas as pd

from app.services.award_tools import infer_award_fields


MAX_ROWS = 5000
SAMPLE_ROW_COUNT = 5


def read_table_file(filename: str, file_bytes: bytes) -> pd.DataFrame:
    """读取用户上传的 CSV 或 Excel 文件。

    这个函数只做一件事：把文件字节变成 DataFrame。
    这样面试时可以清楚说明：上传层负责收文件，这里负责解析数据。
    """

    lower_name = filename.lower()

    if lower_name.endswith(".csv"):
        frame = pd.read_csv(BytesIO(file_bytes))
    elif lower_name.endswith(".xlsx"):
        try:
            # xlsx 文件需要 openpyxl。这里写清楚 engine，方便面试时解释依赖来源。
            frame = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
        except ImportError as exc:
            raise ValueError("读取 XLSX 需要安装 openpyxl，请先运行 pip install openpyxl") from exc
        except Exception as exc:
            raise ValueError(f"XLSX 文件解析失败：{exc}") from exc
    elif lower_name.endswith(".xls"):
        raise ValueError("暂不支持老版 XLS，请另存为 XLSX 后再上传")
    else:
        raise ValueError("只支持 CSV、XLSX 或 XLS 文件")

    if frame.empty:
        raise ValueError("文件中没有可分析的数据")

    # v1 面向简历展示，不处理超大文件。限制行数可以避免本地机器卡住。
    if len(frame) > MAX_ROWS:
        frame = frame.head(MAX_ROWS)

    frame = _promote_real_header(frame)
    frame = _convert_amount_columns(frame)

    return frame


def build_data_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """生成给前端和 Agent 使用的数据概览。"""

    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    text_columns = [name for name in frame.columns if name not in numeric_columns]

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(name) for name in frame.columns],
        "numeric_columns": [str(name) for name in numeric_columns],
        "text_columns": [str(name) for name in text_columns],
        "sample_rows": _safe_records(frame.head(SAMPLE_ROW_COUNT)),
        "all_rows": _safe_records(frame),
        "numeric_summary": _build_numeric_summary(frame, numeric_columns),
        "award_fields": infer_award_fields(frame),
    }


def analyze_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """做一组基础、稳定、可测试的数据分析。"""

    profile = build_data_profile(frame)
    missing_values = frame.isna().sum()

    return {
        "profile": profile,
        "missing_values": {
            str(column): int(count)
            for column, count in missing_values.items()
            if int(count) > 0
        },
        "top_categories": _top_categories(frame, profile["text_columns"]),
    }


def compare_groups(frame: pd.DataFrame, question: str = "") -> dict[str, Any]:
    """按第一个文本列分组，比较第一个数值列的总和与平均值。

    这是一个很常见的业务分析动作：例如按地区比较销售额。
    为了保持简单，v1 不让用户自由选择列，而是自动选择最合适的列。
    """

    profile = build_data_profile(frame)
    text_columns = profile["text_columns"]
    numeric_columns = profile["numeric_columns"]

    if not text_columns or not numeric_columns:
        return {
            "message": "没有同时找到文本列和数值列，暂时无法做分组对比。",
            "groups": [],
        }

    group_column = _find_best_column(text_columns, ["书院", "学院", "地区", "类别", "类型"], question)
    value_column = _find_best_column(numeric_columns, ["奖励额度", "额度", "金额", "销售额", "收入", "成本"], question)
    grouped = (
        frame.groupby(group_column)[value_column]
        .agg(["sum", "mean", "count"])
        .round(2)
        .sort_values("sum", ascending=False)
        .head(10)
    )

    groups = []
    for group_name, row in grouped.iterrows():
        groups.append(
            {
                "group": str(group_name),
                "sum": _to_plain_value(row["sum"]),
                "mean": _to_plain_value(row["mean"]),
                "count": _to_plain_value(row["count"]),
            }
        )

    return {
        "group_column": group_column,
        "value_column": value_column,
        "groups": groups,
    }


def find_outliers(frame: pd.DataFrame) -> dict[str, Any]:
    """用简单 IQR 方法寻找数值列异常值。

    IQR = Q3 - Q1。低于 Q1 - 1.5*IQR 或高于 Q3 + 1.5*IQR 的值会被标记。
    这个方法简单、经典，面试时也容易讲清楚。
    """

    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    result: dict[str, Any] = {}

    for column in numeric_columns:
        values = frame[column].dropna()
        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_rows = frame[(frame[column] < lower) | (frame[column] > upper)]

        if not outlier_rows.empty:
            result[str(column)] = {
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
                "rows": _safe_records(outlier_rows.head(5)),
            }

    return result or {"message": "暂未发现明显异常值。"}


def _build_numeric_summary(frame: pd.DataFrame, numeric_columns: list[str]) -> dict[str, Any]:
    if not numeric_columns:
        return {}

    summary = frame[numeric_columns].describe().round(2)
    return {
        column: {
            stat: _to_plain_value(summary.loc[stat, column])
            for stat in summary.index
        }
        for column in numeric_columns
    }


def _top_categories(frame: pd.DataFrame, text_columns: list[str]) -> dict[str, Any]:
    result = {}

    for column in text_columns[:3]:
        counts = frame[column].astype(str).value_counts().head(5)
        result[str(column)] = {
            str(key): int(value)
            for key, value in counts.items()
        }

    return result


def _promote_real_header(frame: pd.DataFrame) -> pd.DataFrame:
    """如果 Excel 第一行是标题，自动把真正表头提升为列名。"""

    column_names = [str(column) for column in frame.columns]
    unnamed_count = sum(name.startswith("Unnamed") for name in column_names)

    if unnamed_count < max(1, len(column_names) // 2):
        return frame

    for row_index in range(min(5, len(frame))):
        row_values = frame.iloc[row_index].tolist()
        header_values = [str(value).strip() for value in row_values if pd.notna(value)]

        has_serial = "序号" in header_values
        has_college = any("书院" in value or "学院" in value for value in header_values)
        has_amount = any("额度" in value or "金额" in value or "奖励" in value for value in header_values)

        if has_serial and (has_college or has_amount):
            new_columns = []
            for index, value in enumerate(row_values):
                if pd.isna(value) or str(value).strip() == "":
                    new_columns.append(f"未命名列{index + 1}")
                else:
                    new_columns.append(str(value).strip())

            cleaned = frame.iloc[row_index + 1:].copy()
            cleaned.columns = new_columns
            cleaned = cleaned.dropna(how="all")
            return cleaned.reset_index(drop=True)

    return frame


def _convert_amount_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """把“奖励额度/金额”这类列转成数字，方便后续求和画图。"""

    result = frame.copy()

    for column in result.columns:
        column_name = str(column)
        if "额度" not in column_name and "金额" not in column_name:
            continue

        values = (
            result[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("元", "", regex=False)
            .str.strip()
        )
        result[column] = pd.to_numeric(values, errors="coerce")

    return result


def _find_best_column(columns: list[str], keywords: list[str], question: str) -> str:
    for keyword in keywords:
        if keyword in question:
            for column in columns:
                if keyword in str(column):
                    return column

    for keyword in keywords:
        for column in columns:
            if keyword in str(column):
                return column

    return columns[0]


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """把 DataFrame 行转成 JSON 友好的 dict。"""

    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    return [
        {str(key): _to_plain_value(value) for key, value in row.items()}
        for row in records
    ]


def _to_plain_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
