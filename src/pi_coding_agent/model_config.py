"""models.json 不可变快照（对齐 TS core/model-config.ts）。

ModelConfig 是 credentials-blind 的配置快照：

- providers:  provider_id → ProviderOverride；
- models:     所有 provider 的 modelOverrides 按 model_id 展平合并
  （同名 model_id 跨 provider 冲突时，文件中靠后者逐字段覆盖）。

支持 JSON 注释（// 与 /* */），加载失败时记录 error 而不抛出。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# models.json 结构类型
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CostTier:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    input_tokens_above: int = 0


@dataclass(slots=True, frozen=True)
class CostConfig:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    tiers: tuple[CostTier, ...] = ()


@dataclass(slots=True, frozen=True)
class ModelOverride:
    """单个模型的 models.json 覆盖（modelOverrides.<id>）。"""

    name: str | None = None
    aliases: tuple[str, ...] | None = None
    reasoning: bool | None = None
    thinking_level_map: dict[str, str | None] | None = None
    input: tuple[str, ...] | None = None
    cost: CostConfig | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    headers: dict[str, str] | None = None
    compat: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class ModelDefinition:
    """自定义模型定义（providers.<id>.models[]）。"""

    id: str
    name: str | None = None
    aliases: tuple[str, ...] | None = None
    api: str | None = None
    base_url: str | None = None
    reasoning: bool | None = None
    thinking_level_map: dict[str, str | None] | None = None
    input: tuple[str, ...] | None = None
    cost: CostConfig | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    headers: dict[str, str] | None = None
    compat: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class ProviderOverride:
    """单个 provider 的 models.json 配置（providers.<id>）。"""

    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api: str | None = None
    oauth: str | None = None
    headers: dict[str, str] | None = None
    compat: dict[str, Any] | None = None
    auth_header: bool | None = None
    models: tuple[ModelDefinition, ...] = ()
    model_overrides: dict[str, ModelOverride] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """models.json 的不可变快照。"""

    models: dict[str, ModelOverride] = field(default_factory=dict)
    providers: dict[str, ProviderOverride] = field(default_factory=dict)
    error: str | None = None

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    @staticmethod
    async def load(models_json_path: str | Path | None) -> "ModelConfig":
        """读取并校验 models.json；文件不存在时返回空配置。"""
        if not models_json_path:
            return ModelConfig()
        path = Path(models_json_path)
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ModelConfig()
        except OSError as exc:
            return ModelConfig(error=f"Failed to load models.json: {exc}\n\nFile: {path}")

        try:
            parsed = json.loads(strip_json_comments(content))
        except json.JSONDecodeError as exc:
            return ModelConfig(error=f"Failed to parse models.json: {exc}\n\nFile: {path}")

        if not isinstance(parsed, dict):
            return ModelConfig(
                error=f"Invalid models.json schema: root must be an object\n\nFile: {path}"
            )

        providers_raw = parsed.get("providers", {})
        if not isinstance(providers_raw, dict):
            return ModelConfig(error="Invalid models.json schema: providers must be an object")

        providers: dict[str, ProviderOverride] = {}
        flattened: dict[str, ModelOverride] = {}
        errors: list[str] = []

        for provider_id, raw in providers_raw.items():
            if not isinstance(raw, dict):
                errors.append(f"  - {provider_id}: provider config must be an object")
                continue
            try:
                override = _parse_provider_override(provider_id, raw, errors)
            except ValueError as exc:
                errors.append(f"  - {provider_id}: {exc}")
                continue
            providers[provider_id] = override
            for model_id, model_override in override.model_overrides.items():
                previous = flattened.get(model_id)
                flattened[model_id] = _merge_overrides(previous, model_override)

        if errors:
            return ModelConfig(
                models=flattened,
                providers=providers,
                error="Invalid models.json schema:\n" + "\n".join(errors) + f"\n\nFile: {path}",
            )
        return ModelConfig(models=flattened, providers=providers)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_model_override(self, model_id: str) -> ModelOverride | None:
        return self.models.get(model_id)

    def get_provider_override(self, provider_id: str) -> ProviderOverride | None:
        return self.providers.get(provider_id)

    def get_provider_ids(self) -> tuple[str, ...]:
        return tuple(self.providers.keys())

    def get_error(self) -> str | None:
        return self.error


# ---------------------------------------------------------------------------
# 内部解析辅助
# ---------------------------------------------------------------------------


def _parse_cost(data: Any) -> CostConfig:
    if not isinstance(data, dict):
        return CostConfig()
    tiers: list[CostTier] = []
    for tier in data.get("tiers") or []:
        if isinstance(tier, dict):
            tiers.append(
                CostTier(
                    input=_as_float(tier.get("input")),
                    output=_as_float(tier.get("output")),
                    cache_read=_as_float(tier.get("cacheRead")),
                    cache_write=_as_float(tier.get("cacheWrite")),
                    input_tokens_above=int(tier.get("inputTokensAbove", 0) or 0),
                )
            )
    return CostConfig(
        input=_as_float(data.get("input")),
        output=_as_float(data.get("output")),
        cache_read=_as_float(data.get("cacheRead")),
        cache_write=_as_float(data.get("cacheWrite")),
        tiers=tuple(tiers),
    )


def _as_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _as_tuple_strings(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
    return tuple(result)


def _as_str_dict(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(item, str):
            result[str(key)] = item
    return result or None


def _parse_thinking_map(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, str | None] = {}
    for key, item in value.items():
        if isinstance(item, str) or item is None:
            result[str(key)] = item
    return result or None


def _parse_model_override(data: dict[str, Any]) -> ModelOverride:
    return ModelOverride(
        name=data.get("name"),
        reasoning=data.get("reasoning"),
        thinking_level_map=_parse_thinking_map(data.get("thinkingLevelMap")),
        input=_as_tuple_strings(data.get("input")),
        cost=_parse_cost(data.get("cost")),
        context_window=data.get("contextWindow"),
        max_tokens=data.get("maxTokens"),
        headers=_as_str_dict(data.get("headers")),
        compat=data.get("compat") if isinstance(data.get("compat"), dict) else None,
    )


def _parse_model_definition(
    provider_id: str,
    data: dict[str, Any],
    errors: list[str],
) -> ModelDefinition:
    model_id = data.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError('model entry missing "id"')
    context_window = data.get("contextWindow")
    if context_window is not None and context_window <= 0:
        errors.append(f"  - {provider_id}.{model_id}: invalid contextWindow")
    max_tokens = data.get("maxTokens")
    if max_tokens is not None and max_tokens <= 0:
        errors.append(f"  - {provider_id}.{model_id}: invalid maxTokens")
    return ModelDefinition(
        id=model_id,
        name=data.get("name"),
        api=data.get("api"),
        base_url=data.get("baseUrl"),
        reasoning=data.get("reasoning"),
        thinking_level_map=_parse_thinking_map(data.get("thinkingLevelMap")),
        input=_as_tuple_strings(data.get("input")),
        cost=_parse_cost(data.get("cost")),
        context_window=context_window,
        max_tokens=max_tokens,
        headers=_as_str_dict(data.get("headers")),
        compat=data.get("compat") if isinstance(data.get("compat"), dict) else None,
    )


def _parse_provider_override(
    provider_id: str,
    data: dict[str, Any],
    errors: list[str],
) -> ProviderOverride:
    definitions: list[ModelDefinition] = []
    for entry in data.get("models") or []:
        if not isinstance(entry, dict):
            errors.append(f"  - {provider_id}: model entry must be an object")
            continue
        try:
            definitions.append(_parse_model_definition(provider_id, entry, errors))
        except ValueError as exc:
            errors.append(f"  - {provider_id}: {exc}")

    model_overrides: dict[str, ModelOverride] = {}
    raw_overrides = data.get("modelOverrides")
    if isinstance(raw_overrides, dict):
        for model_id, override in raw_overrides.items():
            if isinstance(override, dict):
                model_overrides[str(model_id)] = _parse_model_override(override)

    return ProviderOverride(
        name=data.get("name"),
        base_url=data.get("baseUrl"),
        api_key=data.get("apiKey"),
        api=data.get("api"),
        oauth=data.get("oauth"),
        headers=_as_str_dict(data.get("headers")),
        compat=data.get("compat") if isinstance(data.get("compat"), dict) else None,
        auth_header=data.get("authHeader"),
        models=tuple(definitions),
        model_overrides=model_overrides,
    )


def _merge_overrides(previous: ModelOverride | None, override: ModelOverride) -> ModelOverride:
    """逐字段合并（后者覆盖前者；None 字段表示不覆盖）。"""
    if previous is None:
        return override
    merged = ModelOverride(
        name=override.name if override.name is not None else previous.name,
        reasoning=override.reasoning if override.reasoning is not None else previous.reasoning,
        thinking_level_map=(
            {**(previous.thinking_level_map or {}), **(override.thinking_level_map or {})}
            if override.thinking_level_map is not None
            else previous.thinking_level_map
        ),
        input=override.input if override.input is not None else previous.input,
        cost=override.cost if override.cost is not None else previous.cost,
        context_window=(
            override.context_window
            if override.context_window is not None
            else previous.context_window
        ),
        max_tokens=override.max_tokens if override.max_tokens is not None else previous.max_tokens,
        headers={**(previous.headers or {}), **(override.headers or {})}
        if override.headers is not None
        else previous.headers,
        compat=(
            {**(previous.compat or {}), **(override.compat or {})}
            if override.compat is not None
            else previous.compat
        ),
    )
    return merged


# ---------------------------------------------------------------------------
# JSON 注释剥离（对齐 TS stripJsonComments）
# ---------------------------------------------------------------------------


def strip_json_comments(text: str) -> str:
    """移除 JSON 字符串之外的 // 与 /* */ 注释。"""
    result: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    while index < length:
        char = text[index]
        if in_string:
            result.append(char)
            if char == "\\" and index + 1 < length:
                result.append(text[index + 1])
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == "/" and index + 1 < length:
            next_char = text[index + 1]
            if next_char == "/":
                newline = text.find("\n", index)
                index = length if newline < 0 else newline + 1
                continue
            if next_char == "*":
                end = text.find("*/", index + 2)
                index = length if end < 0 else end + 2
                continue

        result.append(char)
        index += 1
    return "".join(result)


__all__ = [
    "CostTier",
    "CostConfig",
    "ModelOverride",
    "ModelDefinition",
    "ProviderOverride",
    "ModelConfig",
    "strip_json_comments",
]
