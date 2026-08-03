"""自动生成的模型目录（由 src/pi_ai/scripts/generate_models.py 生成，勿手改）。"""

import json
from pathlib import Path

from ..models_store import model_from_dict
from ...types import Model

GENERATED_AT = '2026-08-03T03:25:21.904808+00:00'
MODEL_PROVIDERS: list[str] = ['ant-ling', 'azure-openai-responses', 'deepseek', 'mistral', 'openai', 'openai-codex', 'openrouter', 'vercel-ai-gateway']


def load_generated_models() -> dict[str, list[Model]]:
    """读取 providers/*.json，返回 {provider_id: [Model, ...]}。"""
    data_dir = Path(__file__).parent / "providers"

    result: dict[str, list[Model]] = {}
    for provider_id in MODEL_PROVIDERS:
        path = data_dir / f"{provider_id}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        result[provider_id] = [model_from_dict(m) for m in raw.values()]
    return result


__all__ = ["GENERATED_AT", "MODEL_PROVIDERS", "load_generated_models"]
