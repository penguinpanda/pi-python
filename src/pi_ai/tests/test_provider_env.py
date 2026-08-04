"""get_provider_env_value 测试。"""

from pi_ai.utils.provider_env import get_provider_env_value


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("PI_TEST_VAR", "process")
    assert get_provider_env_value("PI_TEST_VAR") == "process"
    assert get_provider_env_value("PI_TEST_VAR", {"PI_TEST_VAR": "override"}) == "override"


def test_missing_returns_none(monkeypatch):
    monkeypatch.delenv("PI_TEST_MISSING", raising=False)
    assert get_provider_env_value("PI_TEST_MISSING") is None


def test_empty_string_treated_as_unset(monkeypatch):
    # 对齐 TS：空字符串 falsy 回退；纯空白字符串透传原值。
    monkeypatch.setenv("PI_TEST_EMPTY", "")
    assert get_provider_env_value("PI_TEST_EMPTY") is None
    monkeypatch.setenv("PI_TEST_BLANK", "   ")
    assert get_provider_env_value("PI_TEST_BLANK") == "   "


def test_scoped_env_without_process_fallback(monkeypatch):
    monkeypatch.setenv("PI_TEST_VAR", "process")
    assert get_provider_env_value("PI_TEST_VAR", {"PI_TEST_VAR": ""}) == "process"
