"""generate_models.py 测试（fixture 数据，不依赖 TS 仓库）。"""

import json


from pi_ai.models.models_store import model_from_dict

from pi_ai.scripts.generate_models import (
    convert_ts_model,
    load_ts_catalog,
    write_generated,
)


TS_MODEL = {
    "id": "gpt-test",
    "name": "GPT Test",
    "api": "openai-responses",
    "baseUrl": "https://api.openai.com/v1",
    "provider": "openai",
    "reasoning": True,
    "input": ["text", "image"],
    "output": ["text"],
    "cost": {
        "input": 1.0,
        "output": 2.0,
        "cacheRead": 0.1,
        "cacheWrite": 0.0,
        "tiers": [{"inputTokensAbove": 1000, "input": 0.5, "output": 1.0}],
    },
    "contextWindow": 128000,
    "maxTokens": 8192,
    "compat": {"maxTokensField": "max_completion_tokens"},
    "thinkingLevelMap": {"high": "high"},
    "headers": {"X-Test": "1"},
}


def test_convert_ts_model():
    converted = convert_ts_model(TS_MODEL)
    assert converted["id"] == "gpt-test"
    assert converted["base_url"] == "https://api.openai.com/v1"
    assert converted["max_tokens"] == 8192
    assert converted["context_window"] == 128000
    assert converted["compat"]["maxTokensField"] == "max_completion_tokens"
    assert converted["thinking_level_map"] == {"high": "high"}
    assert converted["headers"] == {"X-Test": "1"}
    assert converted["reasoning"] is True
    assert "image" in converted["input"]
    assert converted["cost"]["tiers"][0]["input_tokens_above"] == 1000


def test_load_ts_catalog_from_flat_dir(tmp_path):
    (tmp_path / "openai.json").write_text(json.dumps({"gpt-test": TS_MODEL}), encoding="utf-8")
    catalog = load_ts_catalog(tmp_path)
    assert "openai" in catalog
    assert catalog["openai"]["gpt-test"]["id"] == "gpt-test"


def test_load_ts_catalog_from_providers_dir(tmp_path):
    providers = tmp_path / "providers"
    providers.mkdir()
    (providers / "deepseek.json").write_text(
        json.dumps({"deepseek-chat": {"id": "deepseek-chat", "provider": "deepseek"}}),
        encoding="utf-8",
    )
    (tmp_path / "models.json").write_text("{}", encoding="utf-8")
    catalog = load_ts_catalog(tmp_path)
    assert set(catalog) == {"deepseek"}


def test_write_generated_roundtrip(tmp_path):
    catalog = {"openai": {"gpt-test": TS_MODEL}}
    write_generated(catalog, tmp_path)
    init = tmp_path / "__init__.py"
    provider_json = tmp_path / "providers" / "openai.json"
    assert init.exists()
    assert provider_json.exists()
    raw = json.loads(provider_json.read_text(encoding="utf-8"))
    model = model_from_dict(raw["gpt-test"])
    assert model.id == "gpt-test"
    assert model.base_url == "https://api.openai.com/v1"
    assert model.context_window == 128000
    assert model.compat == {"maxTokensField": "max_completion_tokens"}
    assert model.reasoning is True
    assert "image" in model.input
    assert "MODEL_PROVIDERS" in init.read_text(encoding="utf-8")
