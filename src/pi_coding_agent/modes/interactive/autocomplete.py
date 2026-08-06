"""slash 命令自动补全 provider（对齐 TS autocomplete commands）。"""

from __future__ import annotations

from typing import Any

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


__all__ = ["create_slash_command_provider"]
