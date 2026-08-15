"""注册机制（5.5）——把扩展注册项应用到宿主系统。"""

from __future__ import annotations

import inspect
from typing import Callable

from pi_tui.keybindings import Keybinding


class ExtensionRegistry:
    """聚合扩展注册项并应用到 ModelRuntime / SlashCommandRegistry / KeybindingsManager。"""

    def __init__(
        self,
        runner,
        *,
        model_runtime=None,
        slash_registry=None,
        keybindings_manager=None,
        action_handler_registrar: Callable[[str, Callable], None] | None = None,
    ) -> None:
        self._runner = runner
        self._model_runtime = model_runtime
        self._slash_registry = slash_registry
        self._keybindings_manager = keybindings_manager
        self._action_handler_registrar = action_handler_registrar

    def get_tools(self):
        return self._runner.get_registered_tools()

    def get_commands(self):
        return self._runner.get_registered_commands()

    def get_shortcuts(self):
        return self._runner.get_shortcuts()

    def get_flags(self):
        return self._runner.get_flags()

    def get_providers(self) -> list[tuple[str, dict]]:
        providers: list[tuple[str, dict]] = []
        for extension in self._runner.extensions:
            providers.extend(extension.providers)
        return providers

    def apply(self) -> None:
        """应用 providers + commands + shortcuts。"""
        self._runner.apply_providers()
        self._apply_commands()
        self._apply_shortcuts()

    def _apply_commands(self) -> None:
        if self._slash_registry is None:
            return
        for command in self._runner.get_registered_commands():
            if command.handler is None:
                continue
            handler = self._wrap_command(command.handler)
            self._slash_registry.register(
                command.name,
                handler,
                description=command.description,
                argument_hint=command.argument_hint,
                get_argument_completions=command.get_argument_completions,
            )

    def _wrap_command(self, command_handler):
        async def _handler(_slash_ctx, args: str):
            context = self._runner.create_command_context()
            result = command_handler(context, args)
            if inspect.isawaitable(result):
                return await result
            return result

        return _handler

    def _apply_shortcuts(self) -> None:
        if self._keybindings_manager is None:
            return
        for shortcut in self._runner.get_shortcuts():
            action_id = f"ext.{shortcut.shortcut.replace('+', '_')}"
            self._keybindings_manager.register(
                Keybinding(
                    key=shortcut.shortcut,
                    action_id=action_id,
                    action=action_id.replace(".", "_"),
                    description=shortcut.description or f"Extension shortcut {shortcut.shortcut}",
                )
            )
            # 引擎按 action_<name> 方法分发,扩展没有宿主方法;
            # 把 handler 注册为 action handler,按键才能真正触发。
            if shortcut.handler is not None and self._action_handler_registrar is not None:
                self._action_handler_registrar(action_id, shortcut.handler)


__all__ = ["ExtensionRegistry"]
