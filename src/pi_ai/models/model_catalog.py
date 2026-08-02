"""模型目录展开（对齐 TS model-catalog.ts）。

flattenModelCatalog 把按 api 分组的原始模型表
（provider → api → {modelId: Model}）压平为 {modelId: Model}，
支持 O(1) 按 ID 查找。
"""

from typing import Any, TypeAlias

from .._types import Model

# provider → api → {modelId: Model}
ModelGroups: TypeAlias = dict[str, dict[str, dict[str, Model]]]


def flatten_model_catalog(
    provider_id: str,
    groups: ModelGroups | dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Model]:
    """把 provider 的 API 分组模型表压平为 {modelId: Model}。

    provider_id 仅用于类型标注与文档；实际模型的 provider 字段
    由数据源携带（与 TS 的 ModelCatalog<TProvider> 语义一致）。
    """
    result: dict[str, Model] = {}
    for models_by_id in groups.values():
        for model_id, model in models_by_id.items():
            result[model_id] = model
    return result


__all__ = ["ModelGroups", "flatten_model_catalog"]
