"""ModelConfig + resolve_config_value 单元测试。"""

from __future__ import annotations

from pi_coding_agent.model_config import ModelConfig, strip_json_comments
from pi_coding_agent.resolve_config_value import (
    clear_config_value_cache,
    get_config_value_env_var_names,
    is_command_config_value,
    resolve_config_value,
    resolve_config_value_or_throw,
    resolve_headers,
)


class TestStripJsonComments:
    def test_strips_line_and_block_comments(self):
        text = """
        {
          // line comment
          "a": "http://x", /* block */
          "b": "y" // trailing
        }
        """
        stripped = strip_json_comments(text)
        assert "// line comment" not in stripped
        assert "/* block */" not in stripped
        assert "http://x" in stripped  # URL 中的 // 不受影响


class TestResolveConfigValue:
    def test_literal(self):
        assert resolve_config_value("plain-key") == "plain-key"

    def test_env_var_reference(self, monkeypatch):
        monkeypatch.setenv("TEST_PI_KEY", "sk-env-1")
        assert resolve_config_value("$TEST_PI_KEY") == "sk-env-1"
        assert resolve_config_value("${TEST_PI_KEY}") == "sk-env-1"
        assert resolve_config_value("pre-$TEST_PI_KEY-post") == "pre-sk-env-1-post"

    def test_env_var_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_PI_MISSING", raising=False)
        assert resolve_config_value("$TEST_PI_MISSING") is None
        with __import__("pytest").raises(ValueError):
            resolve_config_value_or_throw("$TEST_PI_MISSING", "test key")

    def test_escapes(self):
        assert resolve_config_value("$$literal") == "$literal"
        assert resolve_config_value("$!literal") == "!literal"

    def test_command_value(self):
        clear_config_value_cache()
        assert is_command_config_value("!echo hello")
        assert resolve_config_value("!echo hello") == "hello"

    def test_env_var_names(self):
        assert get_config_value_env_var_names("a-$FOO-b-${BAR}") == ["FOO", "BAR"]
        assert get_config_value_env_var_names("!cmd") == []

    def test_resolve_headers(self, monkeypatch):
        monkeypatch.setenv("TEST_PI_HEADER", "hv")
        headers = resolve_headers({"X-A": "$TEST_PI_HEADER", "X-B": "literal"})
        assert headers == {"X-A": "hv", "X-B": "literal"}


class TestModelConfigLoad:
    async def test_load_with_comments_and_overrides(self, tmp_path):
        models_json = tmp_path / "models.json"
        models_json.write_text(
            """
            {
              // custom provider config
              "providers": {
                "deepseek": {
                  "baseUrl": "https://api.deepseek.com",
                  "apiKey": "$DEEPSEEK_API_KEY",
                  "headers": { "X-Custom": "v1" },
                  "modelOverrides": {
                    "deepseek-chat": { "reasoning": true, "maxTokens": 8192 }
                  },
                  "models": [
                    { "id": "custom-model", "api": "openai-completions", "baseUrl": "https://x/api" }
                  ]
                }
              }
            }
            """,
            encoding="utf-8",
        )
        config = await ModelConfig.load(models_json)
        assert config.get_error() is None

        provider = config.get_provider_override("deepseek")
        assert provider is not None
        assert provider.base_url == "https://api.deepseek.com"
        assert provider.api_key == "$DEEPSEEK_API_KEY"
        assert provider.headers == {"X-Custom": "v1"}
        assert provider.models[0].id == "custom-model"

        override = config.get_model_override("deepseek-chat")
        assert override is not None
        assert override.reasoning is True
        assert override.max_tokens == 8192

    async def test_missing_file_returns_empty(self, tmp_path):
        config = await ModelConfig.load(tmp_path / "nope.json")
        assert config.get_provider_ids() == ()
        assert config.get_error() is None

    async def test_invalid_json_records_error(self, tmp_path):
        path = tmp_path / "models.json"
        path.write_text("{ not json", encoding="utf-8")
        config = await ModelConfig.load(path)
        assert config.get_error() is not None
        assert "Failed to parse models.json" in config.get_error()

    async def test_none_path(self):
        config = await ModelConfig.load(None)
        assert config.get_error() is None


class TestModelConfigSchemaValidation:
    async def test_custom_model_parses_without_api(self, tmp_path):
        """api/baseUrl 缺失在组合时校验，配置加载仅解析结构。"""
        path = tmp_path / "models.json"
        path.write_text(
            '{"providers": {"acme": {"models": [{"id": "m1"}]}}}',
            encoding="utf-8",
        )
        config = await ModelConfig.load(path)
        assert config.get_error() is None
        provider = config.get_provider_override("acme")
        assert provider is not None
        assert provider.models[0].id == "m1"
