"""CLI 模型解析器（对齐 TS core/model-resolver.ts）。

负责：

- --model / --provider 精确/模糊解析；
- --models 可循环列表（支持 glob 与 `model:thinking` 后缀）；
- 初始模型选择（CLI > scope > 会话恢复 > settings > 第一个可用）。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Literal

from pi_ai import Model
from pi_ai.types.common import ModelThinkingLevel, ThinkingLevel

from .model_runtime import ModelRuntime
from .model_utils import DEFAULT_THINKING_LEVEL

# 合法思考级别（含扩展级别）。
VALID_THINKING_LEVELS: list[ModelThinkingLevel] = [
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]

# 各内置 provider 的默认模型 ID。
default_model_per_provider: dict[str, str] = {
    "openai": "gpt-5-chat-latest",
    "deepseek": "deepseek-v4-flash",
    "qwen": "qwen-plus",
    "faux": "faux-1",
}

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


@dataclass(slots=True)
class ScopedModel:
    """--models 列表中的单个模型（含可选显式思考级别）。"""

    model: Model
    thinking_level: ModelThinkingLevel | None = None


@dataclass(slots=True)
class ParsedModelResult:
    model: Model | None
    thinking_level: ModelThinkingLevel | None
    warning: str | None


@dataclass(slots=True)
class ModelScopeDiagnostic:
    code: Literal["no-match", "invalid-thinking-level"]
    message: str
    pattern: str


@dataclass(slots=True)
class ResolveModelScopeResult:
    scoped_models: list[ScopedModel]
    diagnostics: list[ModelScopeDiagnostic]


@dataclass(slots=True)
class ResolveCliModelResult:
    model: Model | None
    thinking_level: ModelThinkingLevel | None
    warning: str | None
    error: str | None


@dataclass(slots=True)
class InitialModelResult:
    model: Model | None
    thinking_level: ThinkingLevel
    fallback_message: str | None


def is_valid_thinking_level(value: str) -> bool:
    return value in VALID_THINKING_LEVELS


def is_alias(model_id: str) -> bool:
    """模型 ID 是否为别名（无日期后缀）。"""
    if model_id.endswith("-latest"):
        return True
    return not _DATE_SUFFIX_RE.search(model_id)


def find_exact_model_reference_match(
    model_reference: str,
    available_models: list[Model],
) -> Model | None:
    """精确匹配 `provider/model` 或裸 model id（歧义时返回 None）。"""
    trimmed = model_reference.strip()
    if not trimmed:
        return None
    normalized = trimmed.lower()

    canonical = [
        model
        for model in available_models
        if f"{model.provider}/{model.id}".lower() == normalized
    ]
    if len(canonical) == 1:
        return canonical[0]
    if len(canonical) > 1:
        return None

    slash_index = trimmed.find("/")
    if slash_index != -1:
        provider = trimmed[:slash_index].strip()
        model_id = trimmed[slash_index + 1 :].strip()
        if provider and model_id:
            provider_matches = [
                model
                for model in available_models
                if model.provider.lower() == provider.lower()
                and model.id.lower() == model_id.lower()
            ]
            if len(provider_matches) == 1:
                return provider_matches[0]
            if len(provider_matches) > 1:
                return None

    id_matches = [
        model for model in available_models if model.id.lower() == normalized
    ]
    return id_matches[0] if len(id_matches) == 1 else None


def try_match_model(model_pattern: str, available_models: list[Model]) -> Model | None:
    """精确匹配优先，失败后按 id/name 模糊匹配。"""
    exact = find_exact_model_reference_match(model_pattern, available_models)
    if exact is not None:
        return exact

    pattern_lower = model_pattern.lower()
    matches = [
        model
        for model in available_models
        if pattern_lower in model.id.lower() or pattern_lower in (model.name or "").lower()
    ]
    if not matches:
        return None

    aliases = [model for model in matches if is_alias(model.id)]
    if aliases:
        aliases.sort(key=lambda model: model.id, reverse=True)
        return aliases[0]
    matches.sort(key=lambda model: model.id, reverse=True)
    return matches[0]


def parse_model_pattern(
    pattern: str,
    available_models: list[Model],
    *,
    allow_invalid_thinking_level_fallback: bool = True,
) -> ParsedModelResult:
    """解析 `model[:thinking]` 模式。"""
    exact = try_match_model(pattern, available_models)
    if exact is not None:
        return ParsedModelResult(exact, None, None)

    last_colon_index = pattern.rfind(":")
    if last_colon_index == -1:
        return ParsedModelResult(None, None, None)

    prefix = pattern[:last_colon_index]
    suffix = pattern[last_colon_index + 1 :]

    if is_valid_thinking_level(suffix):
        result = parse_model_pattern(
            prefix,
            available_models,
            allow_invalid_thinking_level_fallback=allow_invalid_thinking_level_fallback,
        )
        if result.model is not None:
            return ParsedModelResult(
                result.model,
                None if result.warning else suffix,
                result.warning,
            )
        return result

    if not allow_invalid_thinking_level_fallback:
        # 严格模式：把后缀视为模型 id 的一部分，不解析。
        return ParsedModelResult(None, None, None)

    result = parse_model_pattern(
        prefix,
        available_models,
        allow_invalid_thinking_level_fallback=allow_invalid_thinking_level_fallback,
    )
    if result.model is not None:
        return ParsedModelResult(
            result.model,
            None,
            f'Invalid thinking level "{suffix}" in pattern "{pattern}". Using default instead.',
        )
    return result


def _glob_match(pattern: str, value: str) -> bool:
    """简化 glob：* ? [..] 与 fnmatch 行为一致（nocase）。"""
    import fnmatch

    return fnmatch.fnmatchcase(value.lower(), pattern.lower())


async def resolve_model_scope_with_diagnostics(
    patterns: list[str],
    model_runtime: ModelRuntime,
) -> ResolveModelScopeResult:
    """解析 --models 列表（支持 glob 与 `:thinking` 后缀）。"""
    available_models = list(await model_runtime.get_available())
    scoped: list[ScopedModel] = []
    diagnostics: list[ModelScopeDiagnostic] = []

    def _dedup(model: Model) -> bool:
        return not any(
            existing.model.id == model.id and existing.model.provider == model.provider
            for existing in scoped
        )

    for pattern in patterns:
        has_glob = any(char in pattern for char in "*?[")
        if not has_glob:
            parsed = parse_model_pattern(pattern, available_models)
            if parsed.warning:
                diagnostics.append(
                    ModelScopeDiagnostic(
                        code="invalid-thinking-level",
                        message=parsed.warning,
                        pattern=pattern,
                    )
                )
            if parsed.model is None:
                diagnostics.append(
                    ModelScopeDiagnostic(
                        code="no-match",
                        message=f'No models match pattern "{pattern}"',
                        pattern=pattern,
                    )
                )
            elif _dedup(parsed.model):
                scoped.append(ScopedModel(parsed.model, parsed.thinking_level))
            continue

        # glob 模式：提取可选 `:thinking` 后缀。
        colon_index = pattern.rfind(":")
        glob_pattern = pattern
        thinking_level: ModelThinkingLevel | None = None
        if colon_index != -1:
            suffix = pattern[colon_index + 1 :]
            if is_valid_thinking_level(suffix):
                thinking_level = suffix
                glob_pattern = pattern[:colon_index]

        exact = find_exact_model_reference_match(glob_pattern, available_models)
        if exact is not None:
            if _dedup(exact):
                scoped.append(ScopedModel(exact, thinking_level))
            continue

        matching = [
            model
            for model in available_models
            if _glob_match(glob_pattern, f"{model.provider}/{model.id}")
            or _glob_match(glob_pattern, model.id)
        ]
        if not matching:
            diagnostics.append(
                ModelScopeDiagnostic(
                    code="no-match",
                    message=f'No models match pattern "{pattern}"',
                    pattern=pattern,
                )
            )
            continue
        for model in matching:
            if _dedup(model):
                scoped.append(ScopedModel(model, thinking_level))

    return ResolveModelScopeResult(scoped_models=scoped, diagnostics=diagnostics)


async def resolve_model_scope(
    patterns: list[str],
    model_runtime: ModelRuntime,
) -> list[ScopedModel]:
    """解析 --models 列表；警告打印到 stderr。"""
    result = await resolve_model_scope_with_diagnostics(patterns, model_runtime)
    for diagnostic in result.diagnostics:
        print(f"Warning: {diagnostic.message}", file=sys.stderr)
    return result.scoped_models


def resolve_cli_model(
    *,
    cli_provider: str | None,
    cli_model: str | None,
    cli_thinking: ModelThinkingLevel | None = None,
    model_runtime: ModelRuntime,
) -> ResolveCliModelResult:
    """解析 --model / --provider CLI 参数。"""
    if not cli_model:
        return ResolveCliModelResult(None, None, None, None)

    # 使用全部模型（含未认证），支持 --api-key 首次设置。
    available_models = list(model_runtime.get_models())
    if not available_models:
        return ResolveCliModelResult(
            None,
            None,
            None,
            "No models available. Check your installation or add models to models.json.",
        )

    provider_map = {model.provider.lower(): model.provider for model in available_models}
    provider = (
        provider_map.get(cli_provider.lower()) if cli_provider else None
    )
    if cli_provider and not provider:
        return ResolveCliModelResult(
            None,
            None,
            None,
            f'Unknown provider "{cli_provider}". Use --list-models to see available providers/models.',
        )

    pattern = cli_model
    inferred_provider = False
    if not provider:
        slash_index = cli_model.find("/")
        if slash_index != -1:
            maybe_provider = cli_model[:slash_index]
            canonical = provider_map.get(maybe_provider.lower())
            if canonical:
                provider = canonical
                pattern = cli_model[slash_index + 1 :]
                inferred_provider = True

    if not provider:
        lower = cli_model.lower()
        exact = next(
            (
                model
                for model in available_models
                if model.id.lower() == lower
                or f"{model.provider}/{model.id}".lower() == lower
            ),
            None,
        )
        if exact is not None:
            return ResolveCliModelResult(exact, None, None, None)

    if cli_provider and provider:
        prefix = f"{provider}/"
        if cli_model.lower().startswith(prefix.lower()):
            pattern = cli_model[len(prefix) :]

    candidates = (
        [model for model in available_models if model.provider == provider]
        if provider
        else available_models
    )
    parsed = parse_model_pattern(
        pattern,
        candidates,
        allow_invalid_thinking_level_fallback=False,
    )

    if parsed.model is not None:
        if (
            inferred_provider
            and not model_runtime.has_configured_auth(parsed.model.provider)
        ):
            raw_exact_matches = [
                model
                for model in available_models
                if model.id.lower() == cli_model.lower()
                and not (model.id == parsed.model.id and model.provider == parsed.model.provider)
            ]
            if raw_exact_matches:
                authenticated = [
                    model
                    for model in raw_exact_matches
                    if model_runtime.has_configured_auth(model.provider)
                ]
                if len(authenticated) == 1:
                    return ResolveCliModelResult(authenticated[0], None, None, None)
        return ResolveCliModelResult(parsed.model, parsed.thinking_level, parsed.warning, None)

    if inferred_provider:
        lower = cli_model.lower()
        exact = next(
            (
                model
                for model in available_models
                if model.id.lower() == lower
                or f"{model.provider}/{model.id}".lower() == lower
            ),
            None,
        )
        if exact is not None:
            return ResolveCliModelResult(exact, None, None, None)
        fallback = parse_model_pattern(
            cli_model,
            available_models,
            allow_invalid_thinking_level_fallback=False,
        )
        if fallback.model is not None:
            return ResolveCliModelResult(
                fallback.model, fallback.thinking_level, fallback.warning, None
            )

    if provider:
        # 未知模型 id：为 provider 构造 fallback 模型（自定义模型 id）。
        fallback_pattern = pattern
        fallback_thinking: ModelThinkingLevel | None = None
        if not cli_thinking:
            last_colon = pattern.rfind(":")
            if last_colon != -1:
                suffix = pattern[last_colon + 1 :]
                if is_valid_thinking_level(suffix):
                    fallback_pattern = pattern[:last_colon]
                    fallback_thinking = suffix
        fallback_model = _build_fallback_model(provider, fallback_pattern, available_models)
        if fallback_model is not None:
            requested = cli_thinking or fallback_thinking
            model = (
                _with_reasoning(fallback_model)
                if requested and requested != "off"
                else fallback_model
            )
            warning = parsed.warning or (
                f'Model "{fallback_pattern}" not found for provider "{provider}". '
                "Using custom model id."
            )
            return ResolveCliModelResult(model, fallback_thinking, warning, None)

    display = f"{provider}/{pattern}" if provider else cli_model
    return ResolveCliModelResult(
        None,
        None,
        parsed.warning,
        f'Model "{display}" not found. Use --list-models to see available models.',
    )


def _build_fallback_model(
    provider: str,
    model_id: str,
    available_models: list[Model],
) -> Model | None:
    provider_models = [model for model in available_models if model.provider == provider]
    if not provider_models:
        return None
    default_id = default_model_per_provider.get(provider)
    base_model = (
        next((model for model in provider_models if model.id == default_id), provider_models[0])
        if default_id
        else provider_models[0]
    )
    return Model(
        id=model_id,
        provider=base_model.provider,
        api=base_model.api,
        name=model_id,
        input=list(base_model.input),
        output=list(base_model.output),
        cost=base_model.cost,
        max_tokens=base_model.max_tokens,
        base_url=base_model.base_url,
        context_window=base_model.context_window,
        headers=base_model.headers,
        compat=base_model.compat,
        thinking_level_map=base_model.thinking_level_map,
        reasoning=base_model.reasoning,
    )


def _with_reasoning(model: Model) -> Model:
    return Model(
        id=model.id,
        provider=model.provider,
        api=model.api,
        name=model.name,
        input=list(model.input),
        output=list(model.output),
        cost=model.cost,
        max_tokens=model.max_tokens,
        base_url=model.base_url,
        context_window=model.context_window,
        headers=model.headers,
        compat=model.compat,
        thinking_level_map=model.thinking_level_map,
        reasoning=True,
    )


async def find_initial_model(
    *,
    cli_provider: str | None,
    cli_model: str | None,
    scoped_models: list[ScopedModel],
    is_continuing: bool,
    default_provider: str | None = None,
    default_model_id: str | None = None,
    default_thinking_level: ThinkingLevel | None = None,
    model_runtime: ModelRuntime,
) -> InitialModelResult:
    """寻找初始模型（CLI > scope > settings > 第一个可用）。"""
    if cli_provider and cli_model:
        resolved = resolve_cli_model(
            cli_provider=cli_provider,
            cli_model=cli_model,
            model_runtime=model_runtime,
        )
        if resolved.error:
            raise ValueError(resolved.error)
        if resolved.model is not None:
            return InitialModelResult(resolved.model, DEFAULT_THINKING_LEVEL, None)

    if scoped_models and not is_continuing:
        first = scoped_models[0]
        return InitialModelResult(
            first.model,
            first.thinking_level or default_thinking_level or DEFAULT_THINKING_LEVEL,
            None,
        )

    if default_provider and default_model_id:
        found = model_runtime.get_model(default_provider, default_model_id)
        if found is not None and model_runtime.has_configured_auth(found.provider):
            return InitialModelResult(
                found,
                default_thinking_level or DEFAULT_THINKING_LEVEL,
                None,
            )

    available = list(await model_runtime.get_available())
    if available:
        for provider, default_id in default_model_per_provider.items():
            match = next(
                (
                    model
                    for model in available
                    if model.provider == provider and model.id == default_id
                ),
                None,
            )
            if match is not None:
                return InitialModelResult(match, DEFAULT_THINKING_LEVEL, None)
        return InitialModelResult(available[0], DEFAULT_THINKING_LEVEL, None)

    return InitialModelResult(None, DEFAULT_THINKING_LEVEL, None)


async def restore_model_from_session(
    saved_provider: str,
    saved_model_id: str,
    current_model: Model | None,
    should_print_messages: bool,
    model_runtime: ModelRuntime,
) -> tuple[Model | None, str | None]:
    """从会话恢复模型（不存在 / 未认证时回退）。"""
    restored = model_runtime.get_model(saved_provider, saved_model_id)
    has_auth = (
        model_runtime.has_configured_auth(restored.provider)
        if restored is not None
        else False
    )
    if restored is not None and has_auth:
        if should_print_messages:
            print(f"Restored model: {saved_provider}/{saved_model_id}")
        return restored, None

    reason = "model no longer exists" if restored is None else "no auth configured"
    if should_print_messages:
        print(
            f"Warning: Could not restore model {saved_provider}/{saved_model_id} ({reason})."
        )
    if current_model is not None:
        if should_print_messages:
            print(f"Falling back to: {current_model.provider}/{current_model.id}")
        return (
            current_model,
            f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
            f"Using {current_model.provider}/{current_model.id}.",
        )

    available = list(await model_runtime.get_available())
    if available:
        fallback_model: Model | None = None
        for provider, default_id in default_model_per_provider.items():
            match = next(
                (
                    model
                    for model in available
                    if model.provider == provider and model.id == default_id
                ),
                None,
            )
            if match is not None:
                fallback_model = match
                break
        if fallback_model is None:
            fallback_model = available[0]
        if should_print_messages:
            print(f"Falling back to: {fallback_model.provider}/{fallback_model.id}")
        return (
            fallback_model,
            f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
            f"Using {fallback_model.provider}/{fallback_model.id}.",
        )

    return None, None


__all__ = [
    "ScopedModel",
    "ParsedModelResult",
    "ModelScopeDiagnostic",
    "ResolveModelScopeResult",
    "ResolveCliModelResult",
    "InitialModelResult",
    "default_model_per_provider",
    "VALID_THINKING_LEVELS",
    "is_valid_thinking_level",
    "is_alias",
    "find_exact_model_reference_match",
    "try_match_model",
    "parse_model_pattern",
    "resolve_model_scope",
    "resolve_model_scope_with_diagnostics",
    "resolve_cli_model",
    "find_initial_model",
    "restore_model_from_session",
]
