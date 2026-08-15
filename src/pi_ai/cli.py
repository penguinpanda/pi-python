"""pi_ai.cli — pi-ai 命令行（对齐 TS packages/ai/src/cli.ts）。

命令：
    pi-ai login [provider]   OAuth 登录（未指定 provider 时交互选择）
    pi-ai list               列出可用 OAuth provider
    pi-ai help               用法

凭证默认保存到 ~/.pi/agent/auth.json（与 pi_coding_agent / README 约定一致），
可用环境变量 PI_AUTH_FILE 覆盖路径；
OAuth 引擎见 pi_ai.auth.oauth（builtin_oauth_providers）。

入口：
    python -m pi_ai login openai-codex
    或安装后直接执行 `pi-ai`（见 pyproject [project.scripts]）。
"""

import asyncio
import os
import getpass
import sys

from pathlib import Path
from typing import Sequence

from .auth.types import AuthEvent, AuthPrompt

from .auth import FileCredentialStore
from .auth.oauth import builtin_oauth_providers


def _default_auth_file() -> str:
    """凭证文件路径：PI_AUTH_FILE 优先，否则 ~/.pi/agent/auth.json。

    不沿用 TS 的 CWD auth.json：长期有效的 refresh token 落到任意/共享目录
    会被 git add、同步工具或同目录用户读到。
    """
    env_file = os.environ.get("PI_AUTH_FILE")
    if env_file:
        return env_file
    return str(Path.home() / ".pi" / "agent" / "auth.json")


AUTH_FILE = _default_auth_file()


class _CliAuthInteraction:
    """AuthInteraction 的终端适配（input/print）。"""

    signal: asyncio.Event | None = None

    async def prompt(self, prompt: AuthPrompt) -> str:
        if prompt.get("type") == "select":
            print(f"\n{prompt.get('message', '')}")
            options = prompt.get("options") or []
            for index, option in enumerate(options, 1):
                print(f"  {index}. {option['label']}")
            while True:
                raw = input(f"Enter number (1-{len(options)}): ").strip()
                try:
                    return options[int(raw) - 1]["id"]
                except (ValueError, IndexError):
                    print("Invalid selection.")
        placeholder = prompt.get("placeholder")
        suffix = f" ({placeholder})" if placeholder else ""
        if prompt.get("type") == "secret":
            # API key 等秘密输入不回显（cloudflare 等 provider 使用）。
            return getpass.getpass(f"{prompt.get('message', '')}{suffix}: ")
        return input(f"{prompt.get('message', '')}{suffix}: ")

    def notify(self, event: AuthEvent) -> None:
        if event.get("type") == "auth_url":
            print(f"\nOpen this URL in your browser:\n{event.get('url', '')}")
            instructions = event.get("instructions")
            if instructions:
                print(instructions)
        elif event.get("type") == "device_code":
            print(f"\nOpen this URL in your browser:\n{event.get('verification_uri', '')}")
            print(f"Enter code: {event.get('user_code', '')}")
        elif event.get("type") in ("info", "progress"):
            message = event.get("message")
            if message:
                print(message)


def _auth_store() -> FileCredentialStore:
    """凭证存储：AUTH_FILE 指向的路径（默认 ~/.pi/agent/auth.json）。"""
    return FileCredentialStore(AUTH_FILE)


def _provider_list_text() -> str:
    return "\n".join(
        f"  {provider_id:<20} {name}" for provider_id, name, _flow in builtin_oauth_providers()
    )


def usage() -> str:
    """用法文本（对齐 TS help 输出）。"""
    return (
        "Usage: pi-ai <command> [provider]\n\n"
        "Commands:\n"
        "  login [provider]  Login to an OAuth provider\n"
        "  list              List available providers\n\n"
        "Providers:\n" + _provider_list_text()
    )


async def _list() -> int:
    for provider_id, name, _flow in builtin_oauth_providers():
        print(f"{provider_id:<20} {name}")
    return 0


async def _login(provider_id: str | None) -> int:
    """OAuth 登录；成功后写入 auth.json。"""
    providers = builtin_oauth_providers()
    if provider_id is None:
        print("Select a provider:")
        for index, (_pid, name, _flow) in enumerate(providers, 1):
            print(f"  {index}. {name}")
        while True:
            raw = input(f"Enter number (1-{len(providers)}): ").strip()
            try:
                provider_id = providers[int(raw) - 1][0]
                break
            except (ValueError, IndexError):
                print("Invalid selection.")

    match = next((p for p in providers if p[0] == provider_id), None)
    if match is None:
        raise ValueError(f"Unknown provider: {provider_id}")
    _pid, _name, flow = match

    credential = await flow.login(_CliAuthInteraction())
    store = _auth_store()

    async def _set(_current):
        return credential

    await store.modify(_pid, _set)
    print(f"\nCredentials saved to {AUTH_FILE}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口；返回退出码（0 成功，1 失败）。"""
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args else ""
    if not command or command in ("help", "--help", "-h"):
        print(usage())
        return 0
    try:
        if command == "list":
            return asyncio.run(_list())
        if command == "login":
            provider_id = args[1] if len(args) > 1 else None
            return asyncio.run(_login(provider_id))
        raise ValueError(f"Unknown command: {command}")
    except Exception as exc:  # noqa: BLE001 — CLI 顶层统一报错（对齐 TS main().catch）
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
