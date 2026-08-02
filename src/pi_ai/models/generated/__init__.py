"""自动生成的模型目录（由 scripts/generate_models.py 生成，勿手改）。"""

from pathlib import Path

from ..models_store import model_from_dict
from ..._types import Model

# 生成时间（ISO 8601）；未生成时为 ""。
GENERATED_AT = ""

# 有生成目录的 provider ID 列表。
MODEL_PROVIDERS: list[str] = []


def load_generated_models() -> dict[str, list[Model]]:
    """读取 providers/*.json，返回 {provider_id: [Model, ...]}。

    未生成目录（MODEL_PROVIDERS 为空）时返回空 dict。
    """
    data_dir = Path(__file__).parent / "providers"
    result: dict[str, list[Model]] = {}
    for provider_id in MODEL_PROVIDERS:
        path = data_dir / f"{provider_id}.json"
        if not path.exists():
            continue
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        result[provider_id] = [model_from_dict(m) for m in raw.values()]
    return result


__all__ = ["GENERATED_AT", "MODEL_PROVIDERS", "load_generated_models"]
