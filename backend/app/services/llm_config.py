import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    base_url: str
    model: str


def get_llm_config() -> LLMConfig:
    """读取 LLM 配置。

    优先使用 DeepSeek 配置；保留旧的 LLM_* 环境变量，方便兼容。
    不在代码中保存或打印真实 API Key。
    """

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL") or os.getenv("LLM_MODEL", "deepseek4flash")

    return LLMConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
    )
