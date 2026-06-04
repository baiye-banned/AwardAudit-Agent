from app.services.llm_config import get_llm_config


def test_deepseek_key_has_priority(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("LLM_API_KEY", "old-key")

    config = get_llm_config()

    assert config.api_key == "deepseek-key"
    assert config.model == "deepseek4flash"


def test_legacy_llm_key_still_works(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "old-key")
    monkeypatch.setenv("LLM_MODEL", "custom-model")

    config = get_llm_config()

    assert config.api_key == "old-key"
    assert config.model == "custom-model"
