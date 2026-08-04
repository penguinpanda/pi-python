"""应用级快捷键管理（对齐 TS core/keybindings.ts）。

快捷键以 action id（如 `app.model.cycleForward`）为键，可被
settings.json 的 `keybindings` 节覆盖（单键字符串 / 多键数组 / None 禁用）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Keybinding:
    """单个快捷键绑定。"""

    key: str
    action_id: str
    action: str
    description: str = ""


DEFAULT_APP_KEYBINDINGS: dict[str, Keybinding] = {
    "app.interrupt": Keybinding("escape", "app.interrupt", "interrupt", "Cancel or abort"),
    "app.clear": Keybinding("ctrl+c", "app.clear", "clear", "Clear editor"),
    "app.exit": Keybinding("ctrl+d", "app.exit", "exit", "Exit when editor is empty"),
    "app.thinking.cycle": Keybinding(
        "shift+tab", "app.thinking.cycle", "cycle_thinking", "Cycle thinking level"
    ),
    "app.model.cycleForward": Keybinding(
        "ctrl+p", "app.model.cycleForward", "cycle_model_forward", "Cycle to next model"
    ),
    "app.model.cycleBackward": Keybinding(
        "shift+ctrl+p", "app.model.cycleBackward", "cycle_model_backward", "Cycle to previous model"
    ),
    "app.model.select": Keybinding(
        "ctrl+l", "app.model.select", "select_model", "Open model selector"
    ),
    "app.tools.expand": Keybinding(
        "ctrl+o", "app.tools.expand", "toggle_tools", "Toggle tool output"
    ),
    "app.thinking.toggle": Keybinding(
        "ctrl+t", "app.thinking.toggle", "toggle_thinking", "Toggle thinking blocks"
    ),
    "app.message.followUp": Keybinding(
        "alt+enter", "app.message.followUp", "follow_up", "Queue follow-up message"
    ),
    "app.message.dequeue": Keybinding(
        "alt+up", "app.message.dequeue", "dequeue", "Restore queued messages"
    ),
    # 对齐 TS：Windows 用 alt+v（避免与终端文本粘贴 ctrl+v 冲突），其他平台用 ctrl+v。
    "app.clipboard.pasteImage": Keybinding(
        "alt+v" if os.name == "nt" else "ctrl+v",
        "app.clipboard.pasteImage",
        "paste_image",
        "Paste image from clipboard",
    ),
    "app.session.new": Keybinding(
        "ctrl+n", "app.session.new", "new_session", "Start a new session"
    ),
    "app.session.resume": Keybinding(
        "ctrl+r", "app.session.resume", "resume_session", "Resume a session"
    ),
    "app.message.copy": Keybinding(
        "ctrl+x", "app.message.copy", "copy_last_message", "Copy last assistant message"
    ),
    "app.editor.external": Keybinding(
        "ctrl+g", "app.editor.external", "external_editor", "Open external editor"
    ),
}


class KeybindingsManager:
    """快捷键解析：默认表 + settings 覆盖。"""

    def __init__(
        self,
        defaults: dict[str, Keybinding] | None = None,
        user_bindings: dict[str, Any] | None = None,
    ) -> None:
        self._defaults = dict(defaults or DEFAULT_APP_KEYBINDINGS)
        self._bindings: dict[str, Keybinding] = {}
        self._alt_keys: dict[str, str] = {}
        for action_id, binding in self._defaults.items():
            self._bindings[action_id] = Keybinding(
                key=binding.key,
                action_id=binding.action_id,
                action=binding.action,
                description=binding.description,
            )
        self._by_key: dict[str, str] = {}
        if user_bindings:
            self.set_user_bindings(user_bindings)
        else:
            self._rebuild_index()

    def register(self, binding: Keybinding) -> None:
        """注册/覆盖一个绑定。"""
        self._bindings[binding.action_id] = binding
        self._alt_keys = {
            key: action_id
            for key, action_id in self._alt_keys.items()
            if action_id != binding.action_id
        }
        self._rebuild_index()

    def reset(self) -> None:
        """恢复默认绑定，清除扩展注册的绑定（settings 覆盖由 load_from_settings 重放）。"""
        self._bindings = {}
        self._alt_keys = {}
        for action_id, binding in self._defaults.items():
            self._bindings[action_id] = Keybinding(
                key=binding.key,
                action_id=binding.action_id,
                action=binding.action,
                description=binding.description,
            )
        self._rebuild_index()

    def set_user_bindings(self, user_bindings: dict[str, Any]) -> None:
        """应用 settings keybindings 覆盖（action_id → key/keys/None）。"""
        for action_id, value in user_bindings.items():
            if action_id not in self._bindings:
                continue
            self._alt_keys = {
                key: other for key, other in self._alt_keys.items() if other != action_id
            }
            if value is None:
                continue  # 保持默认（TS 语义：不设置）
            keys: list[str]
            if isinstance(value, str) and value:
                keys = [value]
            elif isinstance(value, list):
                keys = [entry for entry in value if isinstance(entry, str) and entry]
            else:
                keys = []
            if not keys:
                # 空列表 → 禁用该 action。
                self._bindings[action_id].key = ""
                continue
            self._bindings[action_id].key = keys[0]
            if len(keys) > 1:
                self._alt_keys.update({key: action_id for key in keys[1:]})
        self._rebuild_index()

    def load_from_settings(self, settings: dict) -> None:
        """从 settings.json 加载 keybindings 节。"""
        user = settings.get("keybindings")
        if isinstance(user, dict):
            self.set_user_bindings(user)

    def _rebuild_index(self) -> None:
        self._by_key = {}
        for action_id, binding in self._bindings.items():
            if binding.key:
                self._by_key[binding.key] = action_id
        self._by_key.update(self._alt_keys)

    def resolve(self, key: str) -> str | None:
        """按键名解析 action_id。"""
        return self._by_key.get(key)

    def get_action_key(self, action_id: str) -> str | None:
        binding = self._bindings.get(action_id)
        return binding.key if binding and binding.key else None

    def is_enabled(self, action_id: str) -> bool:
        binding = self._bindings.get(action_id)
        return binding is not None and bool(binding.key)

    def all_bindings(self) -> list[Keybinding]:
        """全部启用的绑定（供 Textual BINDINGS 使用）。"""
        return [
            Keybinding(b.key, b.action_id, b.action, b.description)
            for b in self._bindings.values()
            if b.key
        ]


__all__ = ["Keybinding", "KeybindingsManager", "DEFAULT_APP_KEYBINDINGS"]
