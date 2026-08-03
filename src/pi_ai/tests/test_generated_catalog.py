"""models/generated 加载框架 + 实际生成产物测试。

产物由 `python -m pi_ai.scripts.generate_models --source <ts-json-dir>` 生成
（消费 TS generate-models.ts --json-only 输出），勿手改。
"""

from pathlib import Path

from pi_ai.models.generated import (
    GENERATED_AT,
    MODEL_PROVIDERS,
    load_generated_models,
)


def test_generated_at_present():
    """框架就绪后 GENERATED_AT 应为非空 ISO 时间。"""
    assert GENERATED_AT


def test_providers_have_data_files():
    data_dir = Path(__file__).resolve().parents[1] / "models" / "generated" / "providers"
    for provider_id in MODEL_PROVIDERS:
        assert (data_dir / f"{provider_id}.json").exists(), provider_id


def test_load_generated_models_roundtrip():
    catalog = load_generated_models()
    assert set(catalog) == set(MODEL_PROVIDERS)
    assert sum(len(models) for models in catalog.values()) > 0
    for models in catalog.values():
        for model in models:
            assert model.id
            assert model.provider in MODEL_PROVIDERS
            assert model.api
