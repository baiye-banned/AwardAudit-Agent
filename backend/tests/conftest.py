import sys
from pathlib import Path

import pytest


# 测试从项目根目录运行时，Python 默认不知道 backend/app 在哪里。
# 这里把 backend 加到导入路径中，让测试可以直接写 from app.xxx import ...
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(autouse=True)
def disable_real_llm_api(monkeypatch):
    """单测默认不读取本机真实 LLM Key，避免测试访问外部 API。"""

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
