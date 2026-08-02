"""
CLI 入口 — argparse 解析 + 分发到 print_mode。

用法:
    pi -p "read README.md"
    pi --model deepseek-chat -p "what does this code do?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from pi_agent import Agent, AgentOptions
from pi_agent import set_default_stream_fn as set_agent_stream_fn
from pi_ai import create_default_models
from pi_ai.auth.credential_store import FileCredentialStore
from pi_ai.auth.oauth import builtin_oauth_providers

from ._config import get_agent_dir, get_sessions_dir, load_settings
from ._print_mode import run_print_mode
from ._session import AgentSession
from ._session_manager import SessionManager


def main(args: list[str] | None = None) -> int:
    """CLI 主入口（同步包装）。

    Returns:
        退出码: 0=成功, 1=错误
    """
    return asyncio.run(_async_main(args))


async def _async_main(args: list[str] | None = None) -> int:
    """CLI 异步主入口。"""
    # OAuth 子命令：pi login / logout / list（在 argparse 之前拦截，
    # 避免 "login" 被当作位置参数 message 解析）。
    effective_args = args if args is not None else sys.argv[1:]
    if effective_args and effective_args[0] in ("login", "logout", "list"):
        return await _run_auth_command(effective_args)

    parser = _create_parser()
    parsed = parser.parse_args(args)

    # --help / --version 已在 argparse 中处理

    # 确定工作目录
    cwd = str(Path.cwd())

    # 加载配置
    settings = load_settings(cwd)

    # 创建 Models + 设置默认流函数
    models = create_default_models()
    set_agent_stream_fn(models.stream)

    # Ollama 运行时动态发现：用实际安装的模型替换静态列表（失败则保留静态）
    await _refresh_ollama_provider(models)

    # --list-models: 列出所有可用模型后直接退出（支持 --provider 过滤）
    if parsed.list_models:
        return _print_models(models, provider_id=parsed.provider)

    # 解析模型
    model = _resolve_model(models, parsed, settings)

    # 会话管理
    session_manager: SessionManager
    if parsed.no_session:
        session_manager = SessionManager.in_memory(cwd)
    elif parsed.continue_session:
        # 继续最近的会话
        session_manager = _find_latest_session(cwd)
    elif parsed.session:
        session_manager = SessionManager.open(parsed.session, cwd_override=cwd)
    else:
        # 全新会话
        session_manager = SessionManager.create(cwd)

    # 系统提示
    system_prompt = parsed.system_prompt or "You are a helpful coding assistant."
    if parsed.append_system_prompt:
        system_prompt += "\n" + parsed.append_system_prompt

    # 创建 Agent
    agent = Agent(AgentOptions(
        system_prompt=system_prompt,
        model=model,
        # 会话标识透传给 provider，启用提示缓存（prompt_cache_key）
        session_id=session_manager.session_id,
    ))

    # 创建 AgentSession
    session = AgentSession(
        agent=agent,
        session_manager=session_manager,
        cwd=cwd,
        model=model,
    )

    # 运行 print 模式
    message = parsed.message or _read_stdin()
    if not message:
        print("Error: No input message provided. Use -p 'message' or pipe via stdin.", file=sys.stderr)
        return 1

    return await run_print_mode(session, message)


# ---------------------------------------------------------------------------
# OAuth 子命令（pi login / logout / list）
# ---------------------------------------------------------------------------


def _auth_store() -> FileCredentialStore:
    """凭证存储：~/.pi/agent/auth.json（跟随现有配置目录约定）。"""
    return FileCredentialStore(get_agent_dir() / "auth.json")


class _CliAuthInteraction:
    """AuthInteraction 的终端适配（input/print）。"""

    def __init__(self) -> None:
        self.signal = None

    async def prompt(self, prompt) -> str:
        if prompt["type"] == "select":
            print(f"\n{prompt['message']}")
            options = prompt.get("options") or []
            for index, option in enumerate(options, 1):
                description = option.get("description") or ""
                suffix = f" — {description}" if description else ""
                print(f"  {index}. {option['label']}{suffix}")
            while True:
                raw = input(f"Enter number (1-{len(options)}): ").strip()
                try:
                    return options[int(raw) - 1]["id"]
                except (ValueError, IndexError):
                    print("Invalid selection.")
        placeholder = prompt.get("placeholder")
        suffix = f" ({placeholder})" if placeholder else ""
        return input(f"{prompt['message']}{suffix}: ")

    def notify(self, event) -> None:
        if event["type"] == "auth_url":
            print(f"\nOpen this URL in your browser:\n{event['url']}")
            instructions = event.get("instructions")
            if instructions:
                print(instructions)
        elif event["type"] == "device_code":
            print(f"\nOpen this URL in your browser:\n{event['verificationUri']}")
            print(f"Enter code: {event['userCode']}")
        elif event["type"] in ("info", "progress"):
            message = event.get("message")
            if message:
                print(message)


async def _run_auth_command(args: list[str]) -> int:
    command = args[0]
    if command == "list":
        return await _auth_list()
    if command == "login":
        provider_id = args[1] if len(args) > 1 else None
        return await _auth_login(provider_id)
    if command == "logout":
        provider_id = args[1] if len(args) > 1 else None
        return await _auth_logout(provider_id)
    print(f"Unknown auth command: {command}", file=sys.stderr)
    return 1


async def _auth_list() -> int:
    store = _auth_store()
    infos = await store.list()
    logged_in = {info["provider_id"] for info in infos}
    for provider_id, name, _flow in builtin_oauth_providers():
        status = "logged in" if provider_id in logged_in else "not logged in"
        print(f"  {provider_id:<20} {name}  [{status}]")
    return 0


async def _auth_login(provider_id: str | None) -> int:
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
        print(f"Unknown provider: {provider_id}", file=sys.stderr)
        return 1
    _pid, _name, flow = match
    interaction = _CliAuthInteraction()
    try:
        credential = await flow.login(interaction)
    except Exception as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1

    store = _auth_store()

    async def _set(_current):
        return credential

    await store.modify(_pid, _set)
    print(f"\nCredentials saved to {store._path}")
    return 0


async def _auth_logout(provider_id: str | None) -> int:
    if provider_id is None:
        print("Usage: pi logout <provider>", file=sys.stderr)
        return 1
    store = _auth_store()
    await store.delete(provider_id)
    print(f"Logged out: {provider_id}")
    return 0


def _create_parser() -> argparse.ArgumentParser:
    """创建 argparse 解析器（最小核心参数）。"""
    p = argparse.ArgumentParser(
        prog="pi",
        description="Pi Coding Agent — AI-powered coding assistant (minimal core)",
    )

    # 运行模式
    p.add_argument(
        "-p", "--print",
        action="store_true",
        help="Single-shot print mode (default if message is provided)",
    )

    # 模型选择
    p.add_argument("--model", type=str, help="Model ID (e.g., deepseek-chat, gpt-4o)")
    p.add_argument("--provider", type=str, help="Provider ID (e.g., deepseek, openai, ollama, faux)")
    p.add_argument("--list-models", action="store_true",
                   help="List all available models and exit")

    # 系统提示
    p.add_argument("--system-prompt", type=str, help="Override system prompt")
    p.add_argument("--append-system-prompt", type=str, help="Append to system prompt")

    # 会话
    p.add_argument("--session", type=str, help="Path to existing session file to continue")
    p.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                   help="Continue the most recent session")
    p.add_argument("--no-session", action="store_true", help="Don't persist session to disk")

    # 工具控制
    p.add_argument("--tools", type=str, help="Comma-separated tool whitelist (not yet implemented)")
    p.add_argument("--exclude-tools", type=str, help="Comma-separated tool blacklist (not yet implemented)")
    p.add_argument("--no-tools", action="store_true", help="Disable all tools")

    # 版本
    p.add_argument("--version", action="version", version="pi 0.1.0 (minimal core)")

    # 位置参数：用户消息
    p.add_argument("message", nargs="?", type=str, help="User message (optional, can use stdin)")

    return p


def _print_models(models, provider_id: str | None = None) -> int:
    """列出 Provider 及其模型与能力。

    provider_id 非空时只列该 Provider；不存在则返回 1。

    Returns:
        0: 成功；1: provider 不存在
    """
    if provider_id is not None:
        provider = models.get_provider(provider_id)
        if provider is None:
            print(f"Unknown provider: {provider_id}", file=sys.stderr)
            return 1
        providers = [provider]
    else:
        providers = models.get_providers()

    for provider in providers:
        print(f"{provider.name} ({provider.id}):")
        for m in provider.get_models():
            caps = m.capabilities
            labels: list[str] = []
            if caps.reasoning:
                labels.append("thinking")
            if caps.tools:
                labels.append("tools")
            if caps.images:
                labels.append("images")
            print(f"  {m.id:<40} {m.name}  [{', '.join(labels) or 'text'}]")
    return 0


async def _refresh_ollama_provider(models) -> None:
    """尝试用 Ollama /api/tags 动态发现模型并替换 ollama provider。

    经 Models.refresh() 统一编排；失败保留静态/上次列表，不报错。
    """
    await models.refresh()


def _resolve_model(models, parsed, settings: dict):
    """解析模型：CLI > 配置 > 所有 provider 中搜索 > 第一个可用。"""
    from pi_ai import Model

    provider_id: str | None = parsed.provider or settings.get("defaultProvider")
    model_id: str | None = parsed.model or settings.get("defaultModel")

    # 1. provider + model 都指定了 → 精确查找
    if provider_id and model_id:
        model = models.get_model(provider_id, model_id)
        if model:
            return model

    # 2. 只指定了 model → 跨所有 provider 搜索
    if model_id:
        model = models.get_model_by_id(model_id)
        if model:
            return model

    # 3. 只指定了 provider → 返回该 provider 的第一个模型
    if provider_id:
        provider_models = models.get_models(provider_id)
        if provider_models:
            return provider_models[0]

    # 4. 回退到第一个可用模型
    all_models = models.get_models()
    if all_models:
        return all_models[0]

    raise RuntimeError("No models available. Check your provider configuration.")


def _read_stdin() -> str | None:
    """读取 piped stdin（非 TTY 时）。"""
    if sys.stdin.isatty():
        return None
    try:
        return sys.stdin.read().strip()
    except Exception:
        return None


def _find_latest_session(cwd: str) -> SessionManager:
    """在默认会话目录中查找最近修改的会话文件并打开。"""
    sessions_dir = get_sessions_dir()
    if not sessions_dir.exists():
        # 无会话目录，创建新会话
        return SessionManager.create(cwd)

    jsonl_files = sorted(
        sessions_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if jsonl_files:
        return SessionManager.open(jsonl_files[0], cwd_override=cwd)

    # 无会话文件，创建新会话
    return SessionManager.create(cwd)


if __name__ == "__main__":
    sys.exit(main())
