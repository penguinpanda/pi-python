"""SDK 入口：create_agent_session（对齐 TS packages/coding-agent/src/core/sdk.ts）。"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast

from pi_ai import Model
from pi_ai.models.models_store import FileModelsStore, InMemoryModelsStore
from pi_ai.types.common import ModelThinkingLevel, ThinkingLevel
from pi_ai.utils.retry import RetryPolicy
from pi_agent import Agent, AgentOptions, AgentTool, AgentToolResult
from pi_agent.compaction import CompactionSettings

from ._session import AgentSession
from .messages import convert_to_llm
from ._session_manager_v4 import SessionManagerLike, create_session_manager
from .extensions import ExtensionRunner, ToolDefinition
from .model_resolver import ScopedModel, find_initial_model
from .model_runtime import ModelRuntime
from .model_utils import DEFAULT_THINKING_LEVEL, clamp_thinking_level
from .prompt_templates import PromptTemplateLoader
from .resource_loader import DefaultResourceLoader, ResourceLoadResult
from .settings_manager import SettingsManager
from .skills import SkillLoader
from .tools import create_all_tools, create_coding_tools


@dataclass(slots=True)
class CreateAgentSessionOptions:
    """create_agent_session 选项，对齐 TS CreateAgentSessionOptions。"""

    cwd: str = "."
    agent_dir: str | None = None
    model: Model | None = None
    thinking_level: ModelThinkingLevel | None = None
    system_prompt: str | None = None
    session_manager: SessionManagerLike | None = None
    settings_manager: SettingsManager | None = None
    model_runtime: ModelRuntime | None = None
    resource_loader: DefaultResourceLoader | None = None
    tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    no_tools: Literal["all", "builtin"] | None = None
    custom_tools: list[ToolDefinition | AgentTool] | None = None
    scoped_models: list[ScopedModel] | None = None
    turn_retry_policy: RetryPolicy | None = None
    compaction_settings: CompactionSettings | None = None
    skill_loader: SkillLoader | None = None
    template_loader: PromptTemplateLoader | None = None
    extension_runner: ExtensionRunner | None = None
    system_prompt_builder: Any = None
    extension_state: dict | None = None
    session_start_event: dict | None = None


@dataclass(slots=True)
class CreateAgentSessionResult:
    """create_agent_session 的结果。"""

    session: AgentSession
    extensions_result: ResourceLoadResult
    model_fallback_message: str | None = None


class AgentSessionRuntime:
    """host 会话运行时包装（对齐 TS AgentSessionRuntime 核心访问器）。

    session/services/cwd/diagnostics 访问器 + rebind 回调；
    SDK 消费方（RPC server 等）用它统一持有一个可替换的会话。
    """

    def __init__(
        self,
        session: AgentSession,
        cwd: str,
        *,
        diagnostics: list[dict] | None = None,
        model_fallback_message: str | None = None,
        services: Any = None,
    ) -> None:
        self._session = session
        self._cwd = cwd
        self._diagnostics = list(diagnostics or [])
        self._model_fallback_message = model_fallback_message
        self._services = services
        self._rebind: Callable[[AgentSession], Any] | None = None

    @property
    def services(self) -> Any:
        return self._services

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def diagnostics(self) -> list[dict]:
        return list(self._diagnostics)

    @property
    def model_fallback_message(self) -> str | None:
        return self._model_fallback_message

    def set_rebind_session(self, rebind: Callable[[AgentSession], Any] | None) -> None:
        """设置会话重建回调（对齐 TS setRebindSession）。"""
        self._rebind = rebind

    async def rebind(self, session: AgentSession) -> None:
        if self._rebind is not None:
            result = self._rebind(session)
            if inspect.isawaitable(result):
                await result
        self._session = session


def _entry_thinking_level(branch: list[Any]) -> str | None:
    """从会话分支取最后一条 thinking_level_change。"""
    for entry in reversed(branch):
        if isinstance(entry, dict) and entry.get("type") == "thinking_level_change":
            value = entry.get("thinkingLevel") or entry.get("thinking_level")
            return str(value) if value is not None else None
    return None


def _as_agent_tool(tool: ToolDefinition | AgentTool) -> AgentTool:
    """把 ToolDefinition 归一化为 AgentTool（AgentTool 原样返回）。"""
    if isinstance(tool, AgentTool):
        return tool
    original = tool.execute

    async def execute(tool_call_id, params, signal=None, on_update=None, context=None):
        raw = original(tool_call_id, params, signal, on_update, context)
        if inspect.isawaitable(raw):
            raw = await raw
        if isinstance(raw, AgentToolResult):
            return raw
        if isinstance(raw, dict):
            return AgentToolResult(
                content=raw.get("content") or [],
                details=raw.get("details"),
                terminate=raw.get("terminate"),
                usage=raw.get("usage"),
            )
        if raw is None:
            return AgentToolResult(content=[])
        return AgentToolResult(content=[{"type": "text", "text": str(raw)}])

    return AgentTool(
        name=tool.name,
        label=tool.label or tool.name,
        description=tool.description,
        input_schema=tool.parameters or {"type": "object", "properties": {}},
        prompt_snippet=tool.prompt_snippet or None,
        prompt_guidelines=tool.prompt_guidelines,
        execution_mode=cast(Any, tool.execution_mode),
        execute=execute,
    )


async def create_agent_session(
    options: CreateAgentSessionOptions | None = None,
) -> CreateAgentSessionResult:
    """创建一个 AgentSession；未传 model 时从会话/设置/可用模型中选择。"""
    opts = options or CreateAgentSessionOptions()
    cwd = str(Path(opts.cwd).expanduser().resolve())
    agent_dir = str(Path(opts.agent_dir).expanduser().resolve()) if opts.agent_dir else None

    settings_manager = opts.settings_manager or SettingsManager.create(cwd, agent_dir)
    runtime = opts.model_runtime
    if runtime is None:
        from pi_ai import create_default_models

        providers = create_default_models().get_providers()
        runtime = await ModelRuntime.create(
            providers=providers,
            auth_path=str(Path(agent_dir) / "auth.json") if agent_dir else None,
            models_path=str(Path(agent_dir) / "models.json") if agent_dir else None,
            models_store=(
                FileModelsStore(Path(agent_dir) / "models-store.json")
                if agent_dir
                else InMemoryModelsStore()
            ),
            allow_model_network=False,
        )

    resource_loader = opts.resource_loader or DefaultResourceLoader(
        cwd,
        agent_dir,
        settings_manager=settings_manager,
    )
    if opts.resource_loader is None:
        await resource_loader.reload()

    # TS SDK default is a persistent SessionManager.create(cwd).
    session_manager = opts.session_manager or await create_session_manager(cwd)
    existing_messages = session_manager.build_context()
    has_existing_session = bool(existing_messages)

    model = opts.model
    model_fallback_message: str | None = None
    if model is None and has_existing_session:
        saved = session_manager.get_last_model_change()
        if saved is not None:
            restored = runtime.get_model(saved[0], saved[1])
            if restored is not None and runtime.has_configured_auth(restored.provider):
                model = restored
            else:
                model_fallback_message = f"Could not restore model {saved[0]}/{saved[1]}"

    if model is None:
        settings_thinking = settings_manager.get_default_thinking_level()
        initial = await find_initial_model(
            cli_provider=None,
            cli_model=None,
            scoped_models=opts.scoped_models or [],
            is_continuing=has_existing_session,
            default_provider=settings_manager.get_default_provider(),
            default_model_id=settings_manager.get_default_model(),
            default_thinking_level=cast(
                ThinkingLevel | None,
                settings_thinking if settings_thinking != "off" else None,
            ),
            model_runtime=runtime,
        )
        model = initial.model
        if model is None:
            raise ValueError(
                model_fallback_message or "create_agent_session could not find an available model"
            )
        if model_fallback_message is not None:
            model_fallback_message += f". Using {model.provider}/{model.id}"

    branch = session_manager.get_branch()
    if opts.thinking_level is not None:
        raw_thinking_level: ModelThinkingLevel = opts.thinking_level
    elif has_existing_session:
        restored_thinking = _entry_thinking_level(branch)
        raw_thinking_level = cast(
            ModelThinkingLevel,
            restored_thinking
            or settings_manager.get_default_thinking_level()
            or DEFAULT_THINKING_LEVEL,
        )
    else:
        raw_thinking_level = cast(
            ModelThinkingLevel,
            settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL,
        )
    thinking_level = clamp_thinking_level(model, raw_thinking_level)

    default_tools = create_all_tools(cwd)
    default_coding_tools = create_coding_tools(cwd)
    allowed_tool_names: set[str] | None = None
    if opts.tools is not None:
        allowed_tool_names = set(opts.tools)
        selected_tool_names = list(opts.tools)
    elif opts.no_tools == "all":
        # TS noTools="all" disables built-in and custom/extension tools.
        allowed_tool_names = set()
        selected_tool_names = []
    elif opts.no_tools == "builtin":
        selected_tool_names = []
    else:
        selected_tool_names = [tool.name for tool in default_coding_tools]
    excluded_tool_names = set(opts.exclude_tools or []) or None
    builtin_by_name = {tool.name: tool for tool in default_tools}
    tools_override = [
        builtin_by_name[name]
        for name in selected_tool_names
        if name in builtin_by_name
        and (allowed_tool_names is None or name in allowed_tool_names)
        and not (excluded_tool_names and name in excluded_tool_names)
    ]
    if opts.no_tools != "all":
        tools_override.extend(
            tool
            for tool in (_as_agent_tool(item) for item in opts.custom_tools or [])
            if (allowed_tool_names is None or tool.name in allowed_tool_names)
            and not (excluded_tool_names and tool.name in excluded_tool_names)
        )

    extension_runner = opts.extension_runner
    if extension_runner is None:
        extension_runner = ExtensionRunner(
            resource_loader.get_extensions(),
            runtime=resource_loader.get_extension_runtime(),
            cwd=cwd,
            model_runtime=runtime,
        )
        await extension_runner.discover_resources()

    skill_loader = opts.skill_loader or resource_loader.get_skill_loader()
    template_loader = opts.template_loader or resource_loader.get_template_loader()
    default_system_prompt = (
        opts.system_prompt or resource_loader.get_system_prompt() or "You are a helpful assistant."
    )

    def system_prompt_builder() -> str:
        if opts.system_prompt_builder is not None:
            return str(opts.system_prompt_builder())
        return opts.system_prompt or resource_loader.get_system_prompt() or default_system_prompt

    extension_state = opts.extension_state
    if extension_state is None:
        extension_state = {"runner": None, "active_tools": list(tools_override)}
    extension_state.setdefault("active_tools", list(tools_override))

    if has_existing_session and _entry_thinking_level(branch) is None:
        await session_manager.append_thinking_level_change(thinking_level)
    elif not has_existing_session:
        await session_manager.append_model_change(model.provider, model.id)
        await session_manager.append_thinking_level_change(thinking_level)

    retry_settings = settings_manager.get_retry_settings()
    turn_retry_policy = opts.turn_retry_policy
    if turn_retry_policy is None:
        turn_retry_policy = RetryPolicy(
            enabled=bool(retry_settings.get("enabled", True)),
            max_retries=int(retry_settings.get("maxRetries", 3) or 3),
            base_delay_ms=float(retry_settings.get("baseDelayMs", 2000) or 2000),
        )

    provider_retry_settings = settings_manager.get_provider_retry_settings()
    http_idle_timeout_ms = settings_manager.get_http_idle_timeout_ms()
    # SDK 将 timeout=0 视为“不超时”（对齐 TS 使用 max int32）。
    effective_http_timeout_ms = 2147483647 if http_idle_timeout_ms == 0 else http_idle_timeout_ms
    websocket_connect_timeout_ms = settings_manager.get_web_socket_connect_timeout_ms()

    async def stream_with_settings(model, context, options=None):
        stream_options = dict(options or {})
        stream_options.setdefault(
            "timeout_ms", provider_retry_settings.get("timeoutMs") or effective_http_timeout_ms
        )
        if websocket_connect_timeout_ms is not None:
            stream_options.setdefault("websocket_connect_timeout_ms", websocket_connect_timeout_ms)
        if provider_retry_settings.get("maxRetries") is not None:
            stream_options.setdefault("max_retries", provider_retry_settings["maxRetries"])
        if provider_retry_settings.get("maxRetryDelayMs") is not None:
            stream_options.setdefault(
                "max_retry_delay_ms", provider_retry_settings["maxRetryDelayMs"]
            )
        return await runtime.stream(model, context, stream_options)

    def convert_to_llm_with_block_images(messages):
        """settings.images.blockImages=true 时过滤图片内容（对齐 TS SDK）。"""
        converted = convert_to_llm(messages)
        if not settings_manager.get_block_images():
            return converted
        placeholder = "Image reading is disabled."
        filtered: list = []
        for message in converted:
            content = message.get("content")
            if (
                message.get("role") in ("user", "toolResult")
                and isinstance(content, list)
                and any(part.get("type") == "image" for part in content if isinstance(part, dict))
            ):
                new_content = []
                for _index, part in enumerate(content):
                    if isinstance(part, dict) and part.get("type") == "image":
                        if (
                            new_content
                            and isinstance(new_content[-1], dict)
                            and new_content[-1].get("type") == "text"
                            and new_content[-1].get("text") == placeholder
                        ):
                            continue
                        new_content.append({"type": "text", "text": placeholder})
                    else:
                        new_content.append(part)
                message = {**message, "content": new_content}
            filtered.append(message)
        return filtered

    agent = Agent(
        AgentOptions(
            system_prompt=system_prompt_builder(),
            model=model,
            thinking_level=thinking_level,
            tools=tools_override,
            stream_fn=stream_with_settings,
            convert_to_llm=convert_to_llm_with_block_images,
            session_id=session_manager.session_id,
            steering_mode=cast(Any, settings_manager.get_steering_mode()),
            follow_up_mode=cast(Any, settings_manager.get_follow_up_mode()),
            transport=cast(Any, settings_manager.get_transport()),
            thinking_budgets=cast(Any, settings_manager.get_thinking_budgets()),
            retry_policy=turn_retry_policy,
        )
    )

    session = AgentSession(
        agent=agent,
        session_manager=session_manager,
        cwd=cwd,
        model=model,
        tools_override=tools_override,
        turn_retry_policy=turn_retry_policy,
        compaction_settings=opts.compaction_settings,
        model_runtime=runtime,
        settings_manager=settings_manager,
        allowed_tool_names=allowed_tool_names,
        excluded_tool_names=excluded_tool_names,
        scoped_models=opts.scoped_models,
        skill_loader=skill_loader,
        template_loader=template_loader,
        extension_runner=extension_runner,
        system_prompt_builder=system_prompt_builder,
        extension_state=extension_state,
        session_start_event=opts.session_start_event,
    )
    return CreateAgentSessionResult(
        session=session,
        extensions_result=resource_loader.get_result(),
        model_fallback_message=model_fallback_message,
    )


__all__ = [
    "CreateAgentSessionOptions",
    "CreateAgentSessionResult",
    "AgentSessionRuntime",
    "create_agent_session",
]
