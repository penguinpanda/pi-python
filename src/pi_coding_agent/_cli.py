"""
CLI 入口 — argparse 解析 + 分发到 print_mode。

用法:
    pi-python -p "read README.md"
    pi-python --model deepseek-chat -p "what does this code do?"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any

from pi_agent import Agent, AgentOptions
from pi_agent import set_default_stream_fn as set_agent_stream_fn
from pi_ai import Model
from pi_ai.auth.oauth import builtin_oauth_providers

from ._config import ensure_agent_dirs, get_agent_dir, get_sessions_dir
from .extensions import Extension, ExtensionAPI, ExtensionLoader, ExtensionRunner
from .extensions.builtin_llama import create_extension as create_llama_extension
from ._print_mode import run_print_mode, run_print_mode_json
from .file_processor import process_at_files
from .first_time_setup import run_first_time_setup
from .tools import create_all_tools, filter_tools_by_names
from .rpc import run_rpc_mode
from .modes.interactive import run_tui_mode
from ._session import AgentSession
from ._session_manager_v4 import (
    SessionManagerLike,
    create_session_manager,
    in_memory_session_manager,
    list_sessions,
    open_session_manager,
)
from .auth_storage import AuthStorage
from .compaction import compaction_settings_from_config
from .model_resolver import (
    ScopedModel,
    find_initial_model,
    resolve_cli_model,
    resolve_model_scope,
    restore_model_from_session,
)
from .model_runtime import ModelRuntime
from .prompt_templates import PromptTemplateLoader
from .skills import SkillLoader
from .settings_manager import SettingsManager
from .system_prompt import (
    BuildSystemPromptOptions,
    build_system_prompt,
    discover_append_system_prompt_file,
    discover_system_prompt_file,
    load_project_context_files,
    resolve_prompt_input,
    tool_prompt_guidelines_for,
    tool_snippets_for,
)
from .trust import (
    TrustManager,
    has_trust_requiring_project_resources,
    resolve_project_trusted,
)


def main(args: list[str] | None = None) -> int:
    """CLI 主入口（同步包装）。

    Returns:
        退出码: 0=成功, 1=错误
    """
    # 进程标记：子进程据此识别自己在 pi 内（对齐 TS PI_CODING_AGENT）。
    os.environ.setdefault("PI_CODING_AGENT", "true")
    # 下游提前关闭管道（如 `--json | grep -m1`）时按 Unix 惯例静默终止，
    # 避免 Python 默认把 EPIPE 转成 BrokenPipeError traceback。
    # Windows 无 SIGPIPE，由 _print_mode 的 BrokenPipeError 兜底。
    sigpipe = getattr(signal, "SIGPIPE", None)
    if sigpipe is not None:
        try:
            signal.signal(sigpipe, signal.SIG_DFL)
        except (OSError, ValueError):
            pass
    return asyncio.run(_async_main(args))


async def _async_main(args: list[str] | None = None) -> int:
    """CLI 异步主入口。"""
    # OAuth 子命令：pi-python login / logout / list / auth print-*（在 argparse
    # 之前拦截，避免 "login" 被当作位置参数 message 解析）。
    effective_args = args if args is not None else sys.argv[1:]
    if effective_args and effective_args[0] in ("login", "logout", "list", "auth"):
        return await _run_auth_command(effective_args)

    parser = _create_parser()
    parsed = parser.parse_args(args)

    # 补齐 ~/.pi/agent 约定目录（sessions/prompts/skills/extensions/themes/tools/bin）。
    ensure_agent_dirs()

    # --help / --version 已在 argparse 中处理

    # 对齐 TS：无参数且 stdin 为 TTY 时默认进入 TUI，而不是报缺消息。
    if parsed.mode is None and not parsed.message and not parsed.json and sys.stdin.isatty():
        parsed.mode = "tui"

    # 首次启动向导。
    if parsed.setup:
        return await run_first_time_setup(_auth_store())

    # 确定工作目录
    cwd = str(Path.cwd())

    # 加载配置（双层 + 信任感知：先全局，信任决定后加载项目）。
    settings_manager = SettingsManager.create(cwd, project_trusted=False)
    settings = settings_manager.as_dict()

    # 项目信任：启动时解析（TUI 的交互提示由应用内 TrustSelector 承担，
    # 其余模式无 UI 时按 defaultProjectTrust=ask 拒绝并提示）。
    trust_manager = TrustManager()
    needs_trust_decision = (
        has_trust_requiring_project_resources(cwd)
        and trust_manager.is_trusted(cwd) is None
        and settings.get("defaultProjectTrust", "ask") == "ask"
    )
    project_trusted = await resolve_project_trusted(cwd, trust_manager, settings, ui=None)
    # --approve/-a / --no-approve/-na：本次运行的信任覆盖（对齐 TS projectTrustOverride）。
    if parsed.project_trust_override is not None:
        project_trusted = parsed.project_trust_override
    settings_manager.set_project_trusted(project_trusted)
    settings = settings_manager.as_dict()
    if needs_trust_decision and not project_trusted and parsed.mode != "tui":
        print(
            "Warning: project not trusted; .pi settings and resources are ignored. "
            "Use /trust (TUI) or set defaultProjectTrust to trust/block.",
            file=sys.stderr,
        )

    # --preset：从 settings["presets"] 解析命名预设。
    try:
        preset = _resolve_preset(parsed, settings)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if preset is not None:
        if not parsed.model and isinstance(preset.get("model"), str):
            parsed.model = preset["model"]
        if not parsed.provider and isinstance(preset.get("provider"), str):
            parsed.provider = preset["provider"]

    # 创建 ModelRuntime（组合 provider + models.json + auth.json）
    if parsed.offline:
        os.environ["PI_OFFLINE"] = "1"
    runtime = await _create_runtime()
    set_agent_stream_fn(runtime.stream)

    # --api-key：运行时 API key 覆盖（对齐 TS main.ts setRuntimeApiKey）。
    if parsed.api_key:
        if not parsed.provider:
            print(
                "Error: --api-key requires a provider via --provider",
                file=sys.stderr,
            )
            return 1
        await runtime.set_runtime_api_key(parsed.provider, parsed.api_key)

    # --list-models: 列出所有可用模型后直接退出（支持 --provider 过滤与模糊搜索）
    if parsed.list_models is not None:
        return _print_models(
            runtime, provider_id=parsed.provider, search=parsed.list_models or None
        )

    # --export: 导出会话 HTML 后退出（对齐 TS main.ts --export <in> [out]）
    if parsed.export:
        try:
            from .export_html import export_session_to_html

            export_manager = await open_session_manager(parsed.export)
            try:
                out_path = parsed.message[0] if parsed.message else None
                result = export_session_to_html(
                    export_manager,
                    out_path or str(Path(parsed.export).with_suffix(".html")),
                )
                print(f"Exported to: {result}")
                return 0
            finally:
                close = getattr(export_manager, "close", None)
                if close is not None:
                    await close()
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    # 会话 flags 冲突校验（对齐 TS validateForkFlags / validateSessionIdFlags）
    if parsed.fork and (
        parsed.session or parsed.continue_session or parsed.resume or parsed.session_id
    ):
        print(
            "Error: --fork cannot be combined with --session/--continue/--resume/--session-id",
            file=sys.stderr,
        )
        return 1
    if parsed.session_id and (parsed.session or parsed.continue_session or parsed.resume):
        print(
            "Error: --session-id cannot be combined with --session/--continue/--resume",
            file=sys.stderr,
        )
        return 1

    # 会话目录解析：--session-dir > PI_CODING_AGENT_SESSION_DIR > 默认目录。
    sessions_dir: str | Path | None = parsed.session_dir or os.environ.get(
        "PI_CODING_AGENT_SESSION_DIR"
    )

    # 会话管理
    session_manager: SessionManagerLike
    if parsed.no_session:
        session_manager = await in_memory_session_manager(cwd)
    elif parsed.fork:
        source_path = await _resolve_fork_target(parsed.fork, sessions_dir)
        if source_path is None:
            print(f"Error: No session found matching '{parsed.fork}'", file=sys.stderr)
            return 1
        source_manager = await open_session_manager(source_path, cwd_override=cwd)
        try:
            session_manager = await source_manager.fork(
                source_manager.get_leaf_id() or "",
                sessions_dir=sessions_dir,
            )
        finally:
            close = getattr(source_manager, "close", None)
            if close is not None:
                await close()
    elif parsed.resume:
        resumed = await _pick_session_to_resume(cwd, sessions_dir)
        if resumed is None:
            print("No session selected", file=sys.stderr)
            return 0
        session_manager = resumed
    elif parsed.continue_session:
        # 继续最近的会话
        session_manager = await _find_latest_session(cwd, sessions_dir)
    elif parsed.session_id:
        session_manager = await _open_or_create_session_by_id(parsed.session_id, cwd, sessions_dir)
    elif parsed.session:
        session_manager = await open_session_manager(parsed.session, cwd_override=cwd)
    else:
        # 全新会话
        session_manager = await create_session_manager(cwd, sessions_dir=sessions_dir)

    # 解析模型（含 --models 循环列表；继续会话时优先恢复）
    is_continuing = bool(parsed.continue_session or parsed.session or parsed.resume or parsed.fork)
    try:
        model, scoped_models = await _resolve_initial_model(
            runtime, parsed, settings, session_manager, is_continuing
        )
    except (ValueError, RuntimeError) as exc:
        # 模型解析失败（如未知 provider / 无可用模型）应友好报错并退出，
        # 而不是把 Python traceback 抛给用户。
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Phase 4：技能 / 提示模板加载器（全局 + 项目；未信任项目不加载 .pi 资源）。
    project_skills_dir = Path(cwd) / ".pi" / "skills" if project_trusted else None
    project_prompts_dir = Path(cwd) / ".pi" / "prompts" if project_trusted else None
    project_extensions_dir = Path(cwd) / ".pi" / "extensions" if project_trusted else None
    skill_loader = SkillLoader(
        global_dir=get_agent_dir() / "skills",
        project_dir=project_skills_dir,
    )
    template_loader = PromptTemplateLoader(
        global_dir=get_agent_dir() / "prompts",
        project_dir=project_prompts_dir,
    )
    # 启动时扫描一次；否则 /skill: 与 /模板名 永远找不到资源
    # （只有 TUI /reload 会触发加载，print/RPC 模式会静默失败）。
    skill_result = skill_loader.load(
        explicit_paths=parsed.skill,
        only_explicit=parsed.no_skills,
    )
    for diagnostic in skill_result.diagnostics:
        print(f"Warning: {diagnostic.message} ({diagnostic.path})", file=sys.stderr)
    template_result = template_loader.load(
        explicit_paths=parsed.prompt_templates,
        only_explicit=parsed.no_prompt_templates,
    )

    # 系统提示构建器：默认结构化提示（工具说明 + 指南 + 上下文文件 + 技能），
    # /reload 时重新调用（上下文文件与技能会变化）。
    default_tools = create_all_tools(cwd)
    tools_include = (
        [part.strip() for part in parsed.tools.split(",") if part.strip()] if parsed.tools else None
    )
    if preset is not None and parsed.tools is None and isinstance(preset.get("tools"), list):
        tools_include = [str(item) for item in preset["tools"]]
    tools_exclude = (
        [part.strip() for part in parsed.exclude_tools.split(",") if part.strip()]
        if parsed.exclude_tools
        else None
    )
    if parsed.no_tools or parsed.no_builtin_tools:
        selected_tools: list[str] = []
    elif tools_include is not None or tools_exclude:
        selected_tools = [
            tool.name
            for tool in filter_tools_by_names(
                default_tools,
                include=tools_include,
                exclude=tools_exclude,
            )
        ]
    else:
        selected_tools = [tool.name for tool in default_tools]

    extension_state: dict = {
        "runner": None,
        "active_tools": [tool for tool in default_tools if tool.name in selected_tools],
    }

    def system_prompt_builder() -> str:
        custom_prompt = parsed.system_prompt or settings_manager.get_system_prompt()
        if not custom_prompt:
            # 未显式设置时发现 SYSTEM.md（对齐 TS discoverSystemPromptFile：
            # trust-gated 项目 .pi/SYSTEM.md → 全局 agentDir/SYSTEM.md）。
            discovered_system = discover_system_prompt_file(cwd, get_agent_dir(), project_trusted)
            if discovered_system is not None:
                custom_prompt = discovered_system["content"]
        custom_prompt = resolve_prompt_input(custom_prompt, "system prompt")
        append_parts = settings_manager.get_append_system_prompt()
        if parsed.append_system_prompt:
            append_parts.append(parsed.append_system_prompt)
        if preset is not None and isinstance(preset.get("instructions"), str):
            append_parts.append(preset["instructions"])
        if not append_parts:
            # append 源未显式设置时发现 APPEND_SYSTEM.md（对齐 TS）。
            discovered_append = discover_append_system_prompt_file(
                cwd, get_agent_dir(), project_trusted
            )
            if discovered_append is not None:
                append_parts.append(discovered_append["content"])
        append_parts = [
            resolve_prompt_input(part, "append system prompt") or "" for part in append_parts
        ]
        skills = list(skill_loader.all())
        runner = extension_state["runner"]
        if runner is not None:
            skills.extend(runner.get_discovered_skills())
        active_tools = extension_state.get("active_tools") or []
        return build_system_prompt(
            BuildSystemPromptOptions(
                cwd=cwd,
                custom_prompt=custom_prompt,
                selected_tools=selected_tools,
                tool_snippets=tool_snippets_for(active_tools),
                prompt_guidelines=tool_prompt_guidelines_for(active_tools),
                append_system_prompt="\n".join(append_parts) if append_parts else None,
                context_files=(
                    []
                    if parsed.no_context_files
                    else load_project_context_files(cwd, get_agent_dir())
                ),
                skills=skills,
            )
        )

    # Phase 5：扩展（项目 .pi/extensions + 全局 extensions）。
    extension_loader = ExtensionLoader(
        global_dir=get_agent_dir() / "extensions",
        project_dir=project_extensions_dir,
        cwd=cwd,
    )
    extension_result = await extension_loader.load(
        explicit_paths=parsed.extensions,
        discover=not parsed.no_extensions,
    )
    for error in extension_result.errors:
        message = error.error.replace("\n", " ")
        print(f"Warning: {message} ({error.extension_path})", file=sys.stderr)
    builtin_llama = Extension(
        path="<builtin>/llama",
        resolved_path="<builtin>/llama",
        source="builtin",
        hidden=True,
    )
    create_llama_extension(ExtensionAPI(builtin_llama, extension_result.runtime, cwd=cwd))
    extension_result.extensions.append(builtin_llama)
    extension_runner = ExtensionRunner(
        extension_result.extensions,
        runtime=extension_result.runtime,
        cwd=cwd,
        model_runtime=runtime,
    )
    extension_state["runner"] = extension_runner
    # 扩展 flags：加载后补注册为 CLI 参数并重新解析（两段解析）。
    extension_flags = extension_runner.get_flags()
    if extension_flags:
        _register_extension_flags(parser, extension_runner)
        parsed = parser.parse_args(args)
        for flag in extension_flags:
            extension_runner.set_flag_value(flag.name, getattr(parsed, flag.name, None))
    await extension_runner.discover_resources()
    system_prompt = system_prompt_builder()

    # 创建 Agent
    def build_session(
        sm: SessionManagerLike,
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
        agent = Agent(
            AgentOptions(
                system_prompt=system_prompt,
                model=session_model,
                # 会话标识透传给 provider，启用提示缓存（prompt_cache_key）
                session_id=sm.session_id,
            )
        )
        session = AgentSession(
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
            system_prompt_builder=system_prompt_builder,
            extension_state=extension_state,
            restrict_untrusted_tools=bool(settings.get("restrictUntrustedTools")),
        )
        session.project_trusted = project_trusted
        return session

    session = build_session(session_manager, model, scoped_models)
    if preset is not None and isinstance(preset.get("thinking"), str):
        session.set_thinking_level(preset["thinking"])
    if parsed.thinking:
        session.set_thinking_level(parsed.thinking)

    async def session_factory(manager=None) -> AgentSession:
        fresh_manager = manager if manager is not None else await create_session_manager(cwd)
        fresh_model, fresh_scoped = await _resolve_initial_model(
            runtime, parsed, settings, fresh_manager, is_continuing=False
        )
        return build_session(fresh_manager, fresh_model, fresh_scoped)

    async def resume_factory(path: str) -> AgentSession:
        restored_manager = await open_session_manager(path, cwd_override=cwd)
        restored_model, restored_scoped = await _resolve_initial_model(
            runtime, parsed, settings, restored_manager, is_continuing=True
        )
        return build_session(restored_manager, restored_model, restored_scoped)

    async def rebuilder(manager: SessionManagerLike) -> AgentSession:
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

    # TUI 模式：自研引擎交互界面。
    if parsed.mode == "tui":
        startup_resources = {
            "context_files": (
                [] if parsed.no_context_files else load_project_context_files(cwd, get_agent_dir())
            ),
            "skills": [
                {"name": skill.name, "path": skill.file_path} for skill in skill_result.skills
            ],
            "prompts": [
                {"name": template.name, "path": template.file_path} for template in template_result
            ],
            "extensions": [
                {"name": _extension_label(extension), "path": extension.path}
                for extension in extension_result.extensions
            ],
        }
        return await run_tui_mode(
            session,
            runtime,
            session_factory=session_factory,
            resume_factory=resume_factory,
            session_rebuilder=rebuilder,
            settings=settings,
            settings_manager=settings_manager,
            extension_loader=extension_loader,
            trust_manager=trust_manager,
            project_trusted=project_trusted,
            needs_trust_decision=needs_trust_decision,
            no_context_files=parsed.no_context_files,
            startup_resources=startup_resources,
            ui_mode=parsed.tui_mode or None,
            theme_name=parsed.theme or None,
        )

    # 运行 print 模式
    message_parts: list[str] = list(parsed.message)
    if not message_parts:
        stdin_text = _read_stdin()
        if stdin_text:
            message_parts = [stdin_text]
    if not message_parts:
        print(
            "Error: No input message provided. Use -p 'message' or pipe via stdin.", file=sys.stderr
        )
        return 1

    # @file 注入（文本 / 图片）：逐个片段处理，非 @ 片段原样保留（对齐 TS file-processor）。
    images = None
    texts, images = await process_at_files(message_parts, cwd)
    message = "\n\n".join(texts)
    if parsed.json:
        return await run_print_mode_json(session, message, images)
    return await run_print_mode(session, message, images)


# ---------------------------------------------------------------------------
# OAuth 子命令（pi-python login / logout / list）
# ---------------------------------------------------------------------------


def _extension_label(extension) -> str:
    """扩展的紧凑显示名：路径文件名，.py 去掉后缀（对齐 TS 扩展列表）。"""
    path = Path(getattr(extension, "resolved_path", "") or extension.path)
    return path.stem if path.suffix else path.name


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
    if command == "auth":
        if len(args) > 1 and args[1] in ("print-api-key", "print-bearer-token"):
            return await _auth_print(args[1], args[2:])
        print(
            'Unknown auth command. Use "pi auth print-api-key" or "pi auth print-bearer-token".',
            file=sys.stderr,
        )
        return 1
    print(f"Unknown auth command: {command}", file=sys.stderr)
    return 1


async def _auth_print(kind: str, args: list[str]) -> int:
    """pi auth print-api-key / print-bearer-token：单行打印解析后的凭证。

    对齐 TS cli/credential-print.ts：解析 --provider/--model/--min-expiry，
    经 ModelRuntime.getAuth 刷新并打印凭证（bearer token 默认 30 分钟最小有效期）。
    """
    provider: str | None = None
    model: str | None = None
    min_expiry_ms: int | None = None
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--provider" and index + 1 < len(args):
            provider = args[index + 1]
            index += 2
        elif arg == "--model" and index + 1 < len(args):
            model = args[index + 1]
            index += 2
        elif arg == "--min-expiry":
            if kind != "bearer_token":
                print(
                    "Error: --min-expiry is only supported by print-bearer-token", file=sys.stderr
                )
                return 1
            if index + 1 >= len(args):
                print("Error: --min-expiry must use a duration such as 30m or 1h", file=sys.stderr)
                return 1
            match = re.match(r"^(\d+)(ms|s|m|h)$", args[index + 1])
            if not match:
                print("Error: --min-expiry must use a duration such as 30m or 1h", file=sys.stderr)
                return 1
            min_expiry_ms = (
                int(match.group(1))
                * {
                    "ms": 1,
                    "s": 1000,
                    "m": 60_000,
                    "h": 3_600_000,
                }[match.group(2)]
            )
            index += 2
        else:
            print(f"Error: Unknown argument '{arg}'", file=sys.stderr)
            return 1

    if not model:
        print("Error: Credential printing requires --model <model>", file=sys.stderr)
        return 1

    runtime = await _create_runtime()
    resolved = resolve_cli_model(cli_provider=provider, cli_model=model, model_runtime=runtime)
    if resolved.model is None or resolved.error:
        print(
            f"Error: {resolved.error or 'Unable to resolve the requested provider/model'}",
            file=sys.stderr,
        )
        return 1
    target_model = resolved.model

    overrides: Any = {"min_oauth_validity_ms": min_expiry_ms} if min_expiry_ms is not None else None
    auth = await runtime.get_auth(target_model, overrides=overrides)
    if auth is None or not auth.auth.get("api_key"):
        print(
            f"Error: No credential configured for provider '{target_model.provider}'",
            file=sys.stderr,
        )
        return 1
    print(auth.auth["api_key"])
    return 0


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
        print("Usage: pi-python logout <provider>", file=sys.stderr)
        return 1
    store = _auth_store()
    await store.delete(provider_id)
    print(f"Logged out: {provider_id}")
    return 0


def _create_parser() -> argparse.ArgumentParser:
    """创建 argparse 解析器。"""
    p = argparse.ArgumentParser(
        prog="pi-python",
        description="Pi Coding Agent — AI-powered coding assistant",
    )

    # 运行模式
    p.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="Single-shot print mode (default if message is provided)",
    )
    p.add_argument(
        "--mode",
        choices=["print", "rpc", "tui"],
        default=None,
        help="Run mode: print (default), rpc (stdin/stdout JSONL), or tui",
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
    p.add_argument(
        "--model", type=str, help="Model ID (e.g., deepseek-v4-flash, gpt-5-chat-latest)"
    )
    p.add_argument(
        "--provider", type=str, help="Provider ID (e.g., deepseek, openai, ollama, faux)"
    )
    p.add_argument(
        "--models",
        type=str,
        help="Comma-separated model scope list for cycling (e.g., 'deepseek-v4-flash,openai/gpt-5-chat-latest')",
    )
    p.add_argument(
        "--list-models",
        nargs="?",
        const="",
        type=str,
        help="List all available models and exit (optional fuzzy-search argument)",
    )

    # 系统提示
    p.add_argument("--system-prompt", type=str, help="Override system prompt")
    p.add_argument("--append-system-prompt", type=str, help="Append to system prompt")

    # 会话
    p.add_argument("--session", type=str, help="Path to existing session file to continue")
    p.add_argument(
        "-c",
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Continue the most recent session",
    )
    p.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Select a session to resume (interactive picker)",
    )
    p.add_argument(
        "--session-id",
        type=str,
        help="Use exact project session ID, creating it if missing",
    )
    p.add_argument(
        "--fork",
        type=str,
        metavar="PATH_OR_ID",
        help="Fork specific session file or partial UUID into a new session",
    )
    p.add_argument(
        "--session-dir",
        type=str,
        help="Directory for session storage and lookup",
    )
    p.add_argument("--no-session", action="store_true", help="Don't persist session to disk")

    # 工具控制
    p.add_argument("--tools", "-t", type=str, help="Comma-separated tool whitelist")
    p.add_argument("--exclude-tools", "-xt", type=str, help="Comma-separated tool blacklist")
    p.add_argument("--no-tools", "-nt", action="store_true", help="Disable all tools")
    p.add_argument(
        "--no-builtin-tools",
        "-nbt",
        action="store_true",
        help="Disable builtin tools (extension-registered tools still apply)",
    )

    # 资源加载（对齐 TS：--extension / --skill / --prompt-template 可重复）
    p.add_argument(
        "-e",
        "--extension",
        dest="extensions",
        action="append",
        help="Load extension file or directory (repeatable)",
    )
    p.add_argument(
        "--skill",
        action="append",
        help="Load skill file or directory (repeatable; additive even with --no-skills)",
    )
    p.add_argument(
        "--prompt-template",
        dest="prompt_templates",
        action="append",
        help="Load prompt template file or directory (repeatable)",
    )
    p.add_argument("--no-skills", action="store_true", help="Disable skill discovery")
    p.add_argument(
        "--no-context-files",
        "-nc",
        action="store_true",
        help="Disable AGENTS.md and CLAUDE.md discovery and loading",
    )
    p.add_argument(
        "--no-prompt-templates",
        action="store_true",
        help="Disable prompt template discovery",
    )
    p.add_argument("--api-key", type=str, help="Runtime API key override for the resolved provider")
    p.add_argument("--thinking", type=str, help="Thinking level for the session")
    p.add_argument(
        "--no-extensions",
        "-ne",
        action="store_true",
        help="Disable extension discovery (explicit -e paths still load)",
    )
    p.add_argument(
        "--tui-mode",
        choices=["regular", "fullscreen"],
        default=None,
        help="TUI render mode",
    )
    p.add_argument(
        "--theme",
        type=str,
        help="TUI theme name (repeatable to add theme files)",
    )
    p.add_argument("--no-themes", action="store_true", help="Disable theme discovery")
    p.add_argument(
        "-a",
        "--approve",
        dest="project_trust_override",
        action="store_true",
        default=None,
        help="Trust the project for this run (skip trust prompt)",
    )
    p.add_argument(
        "-na",
        "--no-approve",
        dest="project_trust_override",
        action="store_false",
        help="Do not trust the project for this run",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--offline", action="store_true", help="Skip network model refresh")
    p.add_argument(
        "--export",
        type=str,
        metavar="SESSION_FILE",
        help="Export a session to standalone HTML and exit (output path via message arg)",
    )
    p.add_argument(
        "--preset",
        type=str,
        help="Apply a named preset from settings 'presets' (model/provider/tools/instructions/thinking)",
    )

    # 版本
    p.add_argument("--version", action="version", version="pi-python 0.1.0 (minimal core)")

    # 位置参数：用户消息（可多个；@file 注入与普通片段混合，对齐 TS args.ts）
    p.add_argument(
        "message",
        nargs="*",
        type=str,
        help="User message fragments (optional, can use stdin); @file reads file content",
    )

    return p


def _register_extension_flags(parser: argparse.ArgumentParser, runner) -> None:
    """把扩展注册的 flags 补注册为 CLI 参数（两段解析用）。"""
    for flag in runner.get_flags():
        option = f"--{flag.name}"
        if option in parser._option_string_actions:
            continue
        if flag.type == "boolean":
            parser.add_argument(
                option,
                action="store_true",
                default=bool(flag.default),
                help=flag.description or "",
            )
        else:
            parser.add_argument(option, default=flag.default, help=flag.description or "")


def _resolve_preset(parsed, settings: dict) -> dict | None:
    """从 settings['presets'][name] 解析命名预设（对齐 TS preset 机制）。"""
    name = getattr(parsed, "preset", None)
    if not name:
        return None
    presets = settings.get("presets")
    if not isinstance(presets, dict):
        raise ValueError(f'Preset "{name}" not found (settings has no "presets")')
    preset = presets.get(name)
    if not isinstance(preset, dict):
        raise ValueError(f'Preset "{name}" not found')
    return preset


def _allow_model_network() -> bool:
    """PI_OFFLINE=1/true/yes 时禁止模型目录网络刷新。"""
    return os.environ.get("PI_OFFLINE", "").lower() not in ("1", "true", "yes")


def _print_models(
    runtime: ModelRuntime, provider_id: str | None = None, search: str | None = None
) -> int:
    """列出 Provider 及其模型与能力。

    provider_id 非空时只列该 Provider；不存在则返回 1。
    search 非空时按模型 ID/名称模糊过滤（对齐 TS --list-models [search]）。

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

    search_lower = search.strip().lower() if search else None
    for provider in providers:
        models = provider.get_models()
        if search_lower:
            models = [
                m for m in models if search_lower in m.id.lower() or search_lower in m.name.lower()
            ]
        print(f"{provider.name} ({provider.id}):")
        for m in models:
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
    from pi_ai.models.models_store import FileModelsStore

    providers = create_default_models().get_providers()
    runtime = await ModelRuntime.create(
        providers=providers,
        auth_path=str(get_agent_dir() / "auth.json"),
        models_path=str(get_agent_dir() / "models.json"),
        models_store=FileModelsStore(get_agent_dir() / "models-store.json"),
        allow_model_network=_allow_model_network(),
        model_refresh_timeout_ms=15000,
    )
    return runtime


async def _resolve_initial_model(
    runtime: ModelRuntime,
    parsed,
    settings: dict,
    session_manager: SessionManagerLike,
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


async def _find_latest_session(
    cwd: str, sessions_dir: str | Path | None = None
) -> SessionManagerLike:
    """在会话目录中查找最近修改的会话文件并打开。"""
    directory = Path(sessions_dir) if sessions_dir else get_sessions_dir()
    if not directory.exists():
        # 无会话目录，创建新会话
        return await create_session_manager(cwd, sessions_dir=sessions_dir)

    infos = await list_sessions(directory)
    if infos:
        return await open_session_manager(infos[0].path, cwd_override=cwd)

    # 无会话文件，创建新会话
    return await create_session_manager(cwd, sessions_dir=sessions_dir)


async def _pick_session_to_resume(
    cwd: str, sessions_dir: str | Path | None
) -> SessionManagerLike | None:
    """交互选择要恢复的会话（对齐 TS selectSession 的 CLI 简化版）。"""
    directory = Path(sessions_dir) if sessions_dir else get_sessions_dir()
    current = await list_sessions(directory, cwd=cwd)
    if not current:
        print("No saved sessions found", file=sys.stderr)
        return None
    print("Select a session to resume:")
    for index, info in enumerate(current, start=1):
        name = info.name or info.first_message or info.session_id[:8]
        print(f"  {index}. {name}  ({info.path})")
    while True:
        raw = input("Session number (empty to cancel): ").strip()
        if not raw:
            return None
        try:
            choice = int(raw)
        except ValueError:
            continue
        if 1 <= choice <= len(current):
            break
    return await open_session_manager(current[choice - 1].path, cwd_override=cwd)


async def _resolve_fork_target(arg: str, sessions_dir: str | Path | None) -> str | None:
    """--fork 目标解析：文件路径或部分 UUID（对齐 TS resolveSessionArg）。"""
    path = Path(arg).expanduser()
    if path.is_file():
        return str(path)
    directory = Path(sessions_dir) if sessions_dir else get_sessions_dir()
    sessions = await list_sessions(directory)
    matches = [info for info in sessions if info.session_id.startswith(arg)]
    if len(matches) == 1:
        return matches[0].path
    if len(matches) > 1:
        print(f"Error: Multiple sessions match '{arg}'", file=sys.stderr)
    return None


async def _open_or_create_session_by_id(
    session_id: str, cwd: str, sessions_dir: str | Path | None
) -> SessionManagerLike:
    """精确 session ID 打开，缺失则新建（对齐 TS findLocalSessionByExactId → create）。"""
    directory = Path(sessions_dir) if sessions_dir else get_sessions_dir()
    sessions = await list_sessions(directory, cwd=cwd)
    for info in sessions:
        if info.session_id == session_id:
            return await open_session_manager(info.path, cwd_override=cwd)
    print(
        f"Warning: No project session found with id '{session_id}'; creating a new session with that id.",
        file=sys.stderr,
    )
    return await create_session_manager(cwd, sessions_dir=sessions_dir, session_id=session_id)


if __name__ == "__main__":
    sys.exit(main())
