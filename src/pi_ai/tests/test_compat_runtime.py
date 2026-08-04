"""compat_runtime 测试。"""

from pi_ai.api.compat_runtime import (
    compat_value,
    max_tokens_field,
    requires_reasoning_content_on_assistant_messages,
    supports_long_cache_retention,
    supports_openai_grammar_tools,
    supports_strict_mode,
    thinking_format,
)
from pi_ai.types import Model


def _model(compat=None) -> Model:
    return Model(id="m", provider="p", api="openai-completions", compat=compat)


def test_compat_value_defaults():
    assert (
        compat_value(_model(), "maxTokensField", "max_completion_tokens") == "max_completion_tokens"
    )
    assert (
        compat_value(_model({"maxTokensField": "max_tokens"}), "maxTokensField", "x")
        == "max_tokens"
    )


def test_max_tokens_field():
    assert max_tokens_field(_model()) == "max_tokens"
    assert max_tokens_field(_model({"maxTokensField": "max_tokens"})) == "max_tokens"
    assert (
        max_tokens_field(_model({"maxTokensField": "max_completion_tokens"}))
        == "max_completion_tokens"
    )


def test_boolean_compat_defaults():
    model = _model()
    assert supports_long_cache_retention(model) is True
    assert supports_strict_mode(model) is True
    assert supports_openai_grammar_tools(model) is False
    assert requires_reasoning_content_on_assistant_messages(model) is False
    assert thinking_format(model) == "openai"


def test_boolean_compat_overrides():
    model = _model(
        {
            "supportsLongCacheRetention": False,
            "supportsStrictMode": False,
            "supportsOpenAIGrammarTools": True,
            "requiresReasoningContentOnAssistantMessages": True,
            "thinkingFormat": "deepseek",
        }
    )
    assert supports_long_cache_retention(model) is False
    assert supports_strict_mode(model) is False
    assert supports_openai_grammar_tools(model) is True
    assert requires_reasoning_content_on_assistant_messages(model) is True
    assert thinking_format(model) == "deepseek"
