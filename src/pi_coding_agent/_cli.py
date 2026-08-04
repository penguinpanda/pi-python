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
from pi_ai import Model
from pi_ai.auth.oauth import builtin_oauth_providers

from ._config import get_agent_dir, get_sessions_dir, load_settings
from .extensions import ExtensionLoader, ExtensionRunner
from ._print_mode import run_print_mode, run_print_mode_json
from .file_processor import process_at_files
from .first_time_setup import run_first_time_setup
from .tools import filter_tools_by_names
from .rpc import run_rpc_mode
from .modes.interactive import run_tui_mode
from ._session import AgentSession
from ._session_manager import SessionManager
from .auth_storage import AuthStorage
from .compaction import compaction_settings_from_config
from .model_resolver import (
    ScopedModel,
    find_initial_model,
    resolve_model_scope,
    restore_model_from_session,
)
from .model_runtime import ModelRuntime
from .prompt_templates import PromptTemplateLoader
from .skills import SkillLoader


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

    # 首次启动向导。
    if parsed.setup:
        return await run_first_time_setup(_auth_store())

    # 确定工作目录
    cwd = str(Path.cwd())

    # 加载配置
    settings = load_settings(cwd)

    # 创建 ModelRuntime（组合 provider + models.json + auth.json）
    runtime = await _create_runtime()
    set_agent_stream_fn(runtime.stream)

    # --list-models: 列出所有可用模型后直接退出（支持 --provider 过滤）
    if parsed.list_models:
        return _print_models(runtime, provider_id=parsed.provider)

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

    # 解析模型（含 --models 循环列表；继续会话时优先恢复）
    is_continuing = bool(parsed.continue_session or parsed.session)
    try:
        model, scoped_models = await _resolve_initial_model(
            runtime, parsed, settings, session_manager, is_continuing
        )
    except (ValueError, RuntimeError) as exc:
        # 模型解析失败（如未知 provider / 无可用模型）应友好报错并退出，
        # 而不是把 Python traceback 抛给用户。
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # 系统提示
    system_prompt = parsed.system_prompt or "You are a helpful coding assistant."
    if parsed.append_system_prompt:
        system_prompt += "\n" + parsed.append_system_prompt

    # Phase 4：技能 / 提示模板加载器（全局 + 项目）。
    skill_loader = SkillLoader(
        global_dir=get_agent_dir() / "skills",
        project_dir=Path(cwd) / ".pi" / "skills",
    )
    template_loader = PromptTemplateLoader(
        global_dir=get_agent_dir() / "prompts",
        project_dir=Path(cwd) / ".pi" / "prompts",
    )

    # Phase 5：扩展（项目 .pi/extensions + 全局 extensions）。
    extension_loader = ExtensionLoader(
        global_dir=get_agent_dir() / "extensions",
        project_dir=Path(cwd) / ".pi" / "extensions",
        cwd=cwd,
    )
    extension_result = await extension_loader.load()
    extension_runner = ExtensionRunner(
        extension_result.extensions,
        runtime=extension_result.runtime,
        cwd=cwd,
        model_runtime=runtime,
    )

    # 工具白名单 / 黑名单。
    tools_include = (
        [part.strip() for part in parsed.tools.split(",") if part.strip()]
        if parsed.tools
        else None
    )
    tools_exclude = (
        [part.strip() for part in parsed.exclude_tools.split(",") if part.strip()]
        if parsed.exclude_tools
        else None
    )

    # 创建 Agent
    def build_session(
        sm: SessionManager,
        session_model: Model,
        session_scoped: list[ScopedModel],
    ) -> AgentSession:
        tools_override = None
        if parsed.no_tools:
            tools_override = []
        elif tools_include is not None or tools_exclude:
            from .tools import create_all_tools

            tools_override = filter_tools_by_names(
                create_all_tools(cwd),
                include=tools_include,
                exclude=tools_exclude,
            )
        agent = Agent(AgentOptions(
            system_prompt=system_prompt,
            model=session_model,
            # 会话标识透传给 provider，启用提示缓存（prompt_cache_key）
            session_id=sm.session_id,
        ))
        return AgentSession(
            agent=agent,
            session_manager=sm,
            cwd=cwd,
            model=session_model,
            model_runtime=runtime,
            scoped_models=session_scoped,
            skill_loader=skill_loader,
            template_loader=template_loader,
            extension_runner=extension_runner,
            tools_override=tools_override,
            compaction_settings=compaction_settings_from_config(settings),
        )

    session = build_session(session_manager, model, scoped_models)

    async def session_factory() -> AgentSession:
        fresh_manager = SessionManager.create(cwd)
        fresh_model, fresh_scoped = await _resolve_initial_model(
            runtime, parsed, settings, fresh_manager, is_continuing=False
        )
        return build_session(fresh_manager, fresh_model, fresh_scoped)

    async def resume_factory(path: str) -> AgentSession:
        restored_manager = SessionManager.open(path, cwd_override=cwd)
        restored_model, restored_scoped = await _resolve_initial_model(
            runtime, parsed, settings, restored_manager, is_continuing=True
        )
        return build_session(restored_manager, restored_model, restored_scoped)

    async def rebuilder(manager: SessionManager) -> AgentSession:
        model, scoped = await _resolve_initial_model(
            runtime, parsed, settings, manager, is_continuing=True
        )
        return build_session(manager, model, scoped)

    # RPC 模式：stdin/stdout JSONL 无头协议。
    if parsed.mode == "rpc":
        return await run_rpc_mode(
            session,
            runtime,
            session_factory=session_factory,
            session_rebuilder=rebuilder,
        )

    # TUI 模式：Textual 交互界面。
    if parsed.mode == "tui":
        return await run_tui_mode(
            session,
            runtime,
            session_factory=session_factory,
            resume_factory=resume_factory,
            session_rebuilder=rebuilder,
            settings=settings,
            extension_loader=extension_loader,
        )

    # 运行 print 模式
    message = parsed.message or _read_stdin()
    if not message:
        print("Error: No input message provided. Use -p 'message' or pipe via stdin.", file=sys.stderr)
        return 1

    # @file 注入（文本 / 图片）。
    images = None
    if message.startswith("@"):
        texts, images = await process_at_files([message], cwd)
        message = "\n\n".join(texts)
    if parsed.json:
        return await run_print_mode_json(session, message, images)
    return await run_print_mode(session, message, images)


# ---------------------------------------------------------------------------
# OAuth 子命令（pi login / logout / list）
# ---------------------------------------------------------------------------


def _auth_store() -> AuthStorage:
    """凭证存储：~/.pi/agent/auth.json（跟随现有配置目录约定）。"""
    return AuthStorage.create(get_agent_dir() / "auth.json")


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
    path = getattr(store, "path", None) or getattr(store, "_path", None)
    print(f"\nCredentials saved to {path}")
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
    p.add_argument(
        "--mode",
        choices=["print", "rpc", "tui"],
        default=None,
        help="Run mode: print (default), rpc (stdin/stdout JSONL), or tui (Textual)",
    )
    p.add_argument(
        "--setup",
        action="store_true",
        help="Run the first-time setup wizard",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print mode: emit JSON Lines events to stdout",
    )

    # 模型选择
    p.add_argument("--model", type=str, help="Model ID (e.g., deepseek-v4-flash, gpt-5-chat-latest)")
    p.add_argument("--provider", type=str, help="Provider ID (e.g., deepseek, openai, ollama, faux)")
    p.add_argument("--models", type=str,
                   help="Comma-separated model scope list for cycling (e.g., 'deepseek-v4-flash,openai/gpt-5-chat-latest')")
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
    p.add_argument("--tools", type=str, help="Comma-separated tool whitelist")
    p.add_argument("--exclude-tools", type=str, help="Comma-separated tool blacklist")
    p.add_argument("--no-tools", action="store_true", help="Disable all tools")

    # 版本
    p.add_argument("--version", action="version", version="pi 0.1.0 (minimal core)")

    # 位置参数：用户消息
    p.add_argument("message", nargs="?", type=str, help="User message (optional, can use stdin)")

    return p


def _print_models(runtime: ModelRuntime, provider_id: str | None = None) -> int:
    """列出 Provider 及其模型与能力。

    provider_id 非空时只列该 Provider；不存在则返回 1。

    Returns:
        0: 成功；1: provider 不存在
    """
    if provider_id is not None:
        provider = runtime.get_provider(provider_id)
        if provider is None:
            print(f"Unknown provider: {provider_id}", file=sys.stderr)
            return 1
        providers = [provider]
    else:
        providers = runtime.get_providers()

    for provider in providers:
        print(f"{provider.name} ({provider.id}):")
        for m in provider.get_models():
            labels: list[str] = []
            if m.reasoning:
                labels.append("thinking")
            if "image" in m.input:
                labels.append("images")
            if m.deprecated:
                labels.append("deprecated")
            print(f"  {m.id:<40} {m.name}  [{', '.join(labels) or 'text'}]")
    return 0


async def _create_runtime() -> ModelRuntime:
    """创建 ModelRuntime：内置 providers + models.json + auth.json。

    create-time 网络刷新（Ollama 动态发现等）失败不致命：
    错误由 Models.refresh 收集，保留静态/上次缓存列表。
    """
    from pi_ai import create_default_models

    providers = create_default_models().get_providers()
    runtime = await ModelRuntime.create(
        providers=providers,
        auth_path=str(get_agent_dir() / "auth.json"),
        models_path=str(get_agent_dir() / "models.json"),
        allow_model_network=True,
        model_refresh_timeout_ms=15000,
    )
    return runtime


async def _resolve_initial_model(
    runtime: ModelRuntime,
    parsed,
    settings: dict,
    session_manager: SessionManager,
    is_continuing: bool,
) -> tuple[Model, list[ScopedModel]]:
    """解析初始模型 + --models 循环列表。返回 (model, scoped_models)。"""
    scoped_models: list[ScopedModel] = []
    if parsed.models:
        patterns = [part.strip() for part in parsed.models.split(",") if part.strip()]
        scoped_models = await resolve_model_scope(patterns, runtime)

    # 继续会话：优先恢复会话记录的最后模型。
    if is_continuing:
        saved = session_manager.get_last_model_change()
        if saved is not None:
            restored, _fallback = await restore_model_from_session(
                saved[0], saved[1], None, False, runtime
            )
            if restored is not None:
                return restored, scoped_models

    result = await find_initial_model(
        cli_provider=parsed.provider,
        cli_model=parsed.model,
        scoped_models=scoped_models,
        is_continuing=is_continuing,
        default_provider=settings.get("defaultProvider"),
        default_model_id=settings.get("defaultModel"),
        model_runtime=runtime,
    )
    if result.model is None:
        raise RuntimeError("No models available. Check your provider configuration.")
    return result.model, scoped_models


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
