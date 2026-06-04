from io import BytesIO

import pandas as pd

from app.services.data_tools import build_data_profile, read_table_file


def test_read_table_file_loads_csv_bytes():
    csv_bytes = b"city,sales\nBeijing,120\nShanghai,180\n"

    frame = read_table_file("sales.csv", csv_bytes)

    assert list(frame.columns) == ["city", "sales"]
    assert frame.shape == (2, 2)


def test_read_table_file_loads_xlsx_bytes():
    output = BytesIO()
    source = pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "score": [95, 88],
        }
    )
    source.to_excel(output, index=False)

    frame = read_table_file("scores.xlsx", output.getvalue())

    assert list(frame.columns) == ["name", "score"]
    assert frame.shape == (2, 2)
    assert frame.iloc[0]["score"] == 95


def test_read_table_file_uses_real_header_when_excel_has_title_row():
    output = BytesIO()
    source = pd.DataFrame(
        [
            ["2025-2026学年第一学期学生校外获奖奖励公示名单", None, None],
            ["序号", "住宿书院", "奖励额度"],
            [1, "思齐住宿书院", 300],
            [2, "知行住宿书院", 500],
        ]
    )
    source.to_excel(output, index=False, header=False)

    frame = read_table_file("awards.xlsx", output.getvalue())

    assert list(frame.columns) == ["序号", "住宿书院", "奖励额度"]
    assert frame.shape == (2, 3)
    assert frame.iloc[1]["住宿书院"] == "知行住宿书院"


def test_build_data_profile_summarizes_columns_and_sample_rows():
    frame = pd.DataFrame(
        {
            "city": ["Beijing", "Shanghai", "Guangzhou"],
            "sales": [120, 180, 150],
        }
    )

    profile = build_data_profile(frame)

    assert profile["row_count"] == 3
    assert profile["column_count"] == 2
    assert profile["numeric_columns"] == ["sales"]
    assert profile["text_columns"] == ["city"]
    assert profile["sample_rows"][0]["city"] == "Beijing"
    assert len(profile["all_rows"]) == 3
    assert profile["all_rows"][2]["city"] == "Guangzhou"
