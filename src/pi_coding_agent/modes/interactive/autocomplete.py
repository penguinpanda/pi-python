"""slash 命令自动补全 provider（对齐 TS autocomplete commands）。"""

from __future__ import annotations

from typing import Any

from pi_tui.autocomplete import CombinedAutocompleteProvider
from pi_tui.engine.fuzzy import fuzzy_filter

from .slash_commands import SlashCommand

# Python 独有命令不出现在 / 补全菜单（对齐 TS BUILTIN_SLASH_COMMANDS）；
# 仍可手动输入执行。
_PYTHON_ONLY_BUILTIN_NAMES = frozenset(
    {
        "thinking",
        "oauth",
        "extensions",
        "help",
        "input",
        "debug",
        "arminsayshi",
        "dementedelves",
    }
)


def create_slash_command_provider(slash_registry, template_loader=None):
    """构造 slash 命令补全 provider。

    item: {"value": "/name ", "label": description}。插入值带尾空格，
    便于继续输入参数；仅当输入以 `/` 开头且尚未出现空格时返回结果。
    """

    def provider(text: str) -> list[dict[str, Any]]:
        stripped = text.lstrip()
        if not stripped.startswith("/"):
            return []
        if " " in stripped:
            return []
        prefix = stripped[1:]
        items: list[dict[str, Any]] = []
        seen: set[str] = set()

        if slash_registry is not None:
            for command in slash_registry.list():
                name = getattr(command, "name", "")
                if name in _PYTHON_ONLY_BUILTIN_NAMES:
                    continue
                if not name.startswith(prefix):
                    continue
                value = f"/{name} "
                if value in seen:
                    continue
                seen.add(value)
                items.append(
                    {
                        "value": value,
                        "label": getattr(command, "description", "") or name,
                    }
                )

        if template_loader is not None:
            for template in template_loader.all():
                name = getattr(template, "name", "")
                if not name.startswith(prefix):
                    continue
                value = f"/{name} "
                if value in seen:
                    continue
                seen.add(value)
                items.append(
                    {
                        "value": value,
                        "label": getattr(template, "description", "") or name,
                    }
                )
        return items

    return provider


def create_interactive_autocomplete_provider(
    *,
    slash_registry=None,
    template_loader=None,
    extension_runner=None,
    skill_loader=None,
    settings_manager=None,
    model_runtime=None,
    session=None,
    base_path: str | None = None,
    fd_path: str | None = None,
) -> CombinedAutocompleteProvider:
    """构造 Interactive Mode 的统一 autocomplete provider。

    候选来源顺序对齐 TS createBaseAutocompleteProvider：
    builtin commands → templates → extension commands → skills；
    /model 与 /login 使用参数补全；路径补全由 pi_tui provider 内置。
    """
    commands: list[SlashCommand] = []
    seen: set[str] = set()
    providers: list = []

    if slash_registry is not None:
        for command in slash_registry.list():
            name = getattr(command, "name", "")
            if name in _PYTHON_ONLY_BUILTIN_NAMES or name in seen:
                continue
            seen.add(name)
            wrapped = SlashCommand(
                name=name,
                handler=getattr(command, "handler", lambda _ctx, _args: None),
                description=getattr(command, "description", "") or name,
                argument_hint=getattr(command, "argument_hint", None),
            )
            if name == "model":
                wrapped.get_argument_completions = _model_argument_completions(
                    session,
                    model_runtime,
                )
            elif name == "login":
                wrapped.get_argument_completions = _login_argument_completions(model_runtime)
            commands.append(wrapped)

    if template_loader is not None:
        for template in template_loader.all():
            name = getattr(template, "name", "")
            if name in seen:
                continue
            seen.add(name)
            commands.append(
                SlashCommand(
                    name=name,
                    handler=lambda _ctx, _args: None,
                    description=getattr(template, "description", "") or name,
                    argument_hint=getattr(template, "argument_hint", None),
                )
            )

    if extension_runner is not None:
        getter = getattr(extension_runner, "get_autocomplete", None)
        if getter is not None:
            providers.extend(getter())
        for command in extension_runner.get_registered_commands():
            name = getattr(command, "name", "")
            if name in seen:
                continue
            seen.add(name)
            commands.append(
                SlashCommand(
                    name=name,
                    handler=getattr(command, "handler", lambda _ctx, _args: None),
                    description=getattr(command, "description", "") or name,
                    argument_hint=getattr(command, "argument_hint", None),
                    get_argument_completions=getattr(
                        command,
                        "get_argument_completions",
                        None,
                    ),
                )
            )

    if _skills_enabled(settings_manager):
        for skill in _collect_skills(skill_loader, extension_runner):
            name = f"skill:{getattr(skill, 'name', '')}"
            if name in seen:
                continue
            seen.add(name)
            commands.append(
                SlashCommand(
                    name=name,
                    handler=lambda _ctx, _args: None,
                    description=getattr(skill, "description", "") or name,
                )
            )

    return CombinedAutocompleteProvider(
        providers=providers,
        commands=commands,
        base_path=base_path or ".",
        fd_path=fd_path,
    )


def _model_argument_completions(session, model_runtime):
    async def complete(prefix: str) -> list[dict[str, Any]]:
        models = []
        if session is not None and getattr(session, "scoped_models", None):
            models = [scoped.model for scoped in session.scoped_models]
        elif model_runtime is not None:
            models = model_runtime.get_available_snapshot()
        items = [
            {
                "value": f"{model.provider}/{model.id}",
                "label": model.id,
                "description": _model_description(model),
                "source": "model",
            }
            for model in models
        ]
        return fuzzy_filter(
            items,
            prefix,
            lambda item: _model_search_text(item),
        )

    return complete


def _model_description(model) -> str:
    parts = [model.provider]
    name = getattr(model, "name", "") or ""
    if name:
        parts.append(name)
    return " · ".join(parts)


def _model_search_text(item: dict[str, Any]) -> str:
    provider, separator, name = str(item.get("description", "")).partition(" · ")
    model_id = str(item.get("label", ""))
    suffix = f" {name}" if separator and name else ""
    return f"{provider} {provider}/{model_id} {provider} {model_id}{suffix}"


def _login_argument_completions(model_runtime):
    async def complete(prefix: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        providers = model_runtime.get_providers() if model_runtime is not None else []
        for provider in providers:
            auth = getattr(provider, "auth", None)
            if auth is None:
                continue
            auth_type = "oauth" if hasattr(auth, "oauth") else "api_key"
            status = ""
            if model_runtime is not None:
                try:
                    check = await model_runtime.check_auth(provider.id)
                except Exception:
                    check = None
                if check:
                    status = f"configured ({check.get('source', '')})"
            description = " ".join(part for part in (auth_type, status) if part)
            items.append(
                {
                    "value": provider.id,
                    "label": provider.id,
                    "description": description,
                    "source": "login",
                }
            )
        if not items:
            from pi_ai.auth.oauth import builtin_oauth_providers

            for provider_id, name, _flow in builtin_oauth_providers():
                items.append(
                    {
                        "value": provider_id,
                        "label": provider_id,
                        "description": f"oauth {name}",
                        "source": "login",
                    }
                )
        return fuzzy_filter(
            items,
            prefix,
            lambda item: f"{item['label']} {item['description']} {item['value']}",
        )

    return complete


def _skills_enabled(settings_manager) -> bool:
    if settings_manager is None:
        return True
    getter = getattr(settings_manager, "get_enable_skill_commands", None)
    return bool(getter()) if getter is not None else True


def _collect_skills(skill_loader, extension_runner) -> list[Any]:
    skills: dict[str, Any] = {}
    if skill_loader is not None:
        for skill in skill_loader.all():
            skills.setdefault(getattr(skill, "name", ""), skill)
    if extension_runner is not None:
        getter = getattr(extension_runner, "get_discovered_skills", None)
        if getter is not None:
            for skill in getter():
                skills.setdefault(getattr(skill, "name", ""), skill)
    return list(skills.values())


__all__ = [
    "CombinedAutocompleteProvider",
    "create_interactive_autocomplete_provider",
    "create_slash_command_provider",
]
