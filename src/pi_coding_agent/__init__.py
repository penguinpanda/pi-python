"""
pi-coding-agent  CLI 编码代理

依赖 pi_agent + pi_ai，提供 print 模式的编码代理 CLI。

用法:
    from pi_coding_agent import AgentSession, create_all_tools, run_print_mode, main

    # 编程方式
    session = AgentSession(agent, session_manager, cwd, model)
    await run_print_mode(session, "read README.md")

    # CLI 方式
    # $ pi -p "read README.md"
"""

from ._session import AgentSession
from ._types import AgentSessionConfig, PrintModeOptions
from ._session_manager import SessionInfo, SessionManager, SessionTreeNode
from ._cli import main
from ._print_mode import run_print_mode
from ._config import ensure_agent_dirs, get_agent_dir, get_sessions_dir, load_settings
from ._config import get_prompts_dir, get_skills_dir
from .extensions import (
    EventBus,
    Extension,
    ExtensionAPI,
    ExtensionCommandContext,
    ExtensionContext,
    ExtensionFlag,
    ExtensionLoader,
    ExtensionRegistry,
    ExtensionRunner,
    ExtensionShortcut,
    NoopUIContext,
    RegisteredCommand,
    ToolDefinition,
    UIContext,
)
from .export_html import export_session_to_html
from .file_processor import process_at_files
from .first_time_setup import run_first_time_setup, run_first_time_setup_sync
from .trust import TrustManager, project_has_local_resources, resolve_project_trusted
from .auth_storage import AuthStorage, FileAuthStorageBackend
from .model_config import ModelConfig, ModelOverride, ProviderOverride
from .model_registry import ModelRegistry
from .model_runtime import ModelRuntime
from .prompt_templates import PromptTemplate, PromptTemplateLoader, parse_command_args
from .skills import (
    LoadSkillsResult,
    ResourceDiagnostic,
    Skill,
    SkillLoader,
    format_skills_for_prompt,
)
from .model_utils import (
    DEFAULT_THINKING_LEVEL,
    THINKING_LEVELS,
    clamp_thinking_level,
    get_supported_thinking_levels,
)
from .resolve_config_value import resolve_config_value
from .rpc import RpcClient, RpcMessageHandler, RpcUiContext, run_rpc_mode
from .modes.interactive import PiTuiApp, SlashCommandRegistry, SlashContext, run_tui_mode
from pi_tui import Keybinding, KeybindingsManager, Theme, ThemeLoader
from .tools import (
    create_all_tools,
    create_coding_tools,
    create_readonly_tools,
    create_read_tool,
    create_write_tool,
    create_edit_tool,
    create_bash_tool,
    create_grep_tool,
    create_find_tool,
    create_ls_tool,
)

__all__ = [
    # Core
    "AgentSession",
    "AgentSessionConfig",
    "PrintModeOptions",
    "SessionManager",
    "SessionTreeNode",
    "SessionInfo",
    # Modes
    "main",
    "run_print_mode",
    # Config
    "get_agent_dir",
    "get_sessions_dir",
    "load_settings",
    "ensure_agent_dirs",
    "get_skills_dir",
    "get_prompts_dir",
    # Model runtime (Phase 1)
    "ModelRuntime",
    "ModelConfig",
    "ModelOverride",
    "ProviderOverride",
    "ModelRegistry",
    "AuthStorage",
    "FileAuthStorageBackend",
    "THINKING_LEVELS",
    "DEFAULT_THINKING_LEVEL",
    "get_supported_thinking_levels",
    "clamp_thinking_level",
    "resolve_config_value",
    # RPC mode (Phase 2)
    "RpcClient",
    "RpcMessageHandler",
    "RpcUiContext",
    "run_rpc_mode",
    # TUI mode (Phase 3)
    "PiTuiApp",
    "run_tui_mode",
    "Keybinding",
    "KeybindingsManager",
    "SlashCommandRegistry",
    "SlashContext",
    "Theme",
    "ThemeLoader",
    # Skills + Prompt Templates (Phase 4)
    "Skill",
    "SkillLoader",
    "LoadSkillsResult",
    "ResourceDiagnostic",
    "format_skills_for_prompt",
    "PromptTemplate",
    "PromptTemplateLoader",
    "parse_command_args",
    # Extensions (Phase 5)
    "ExtensionLoader",
    "ExtensionRunner",
    "ExtensionRegistry",
    "Extension",
    "ExtensionAPI",
    "ExtensionContext",
    "ExtensionCommandContext",
    "ExtensionFlag",
    "ExtensionShortcut",
    "RegisteredCommand",
    "ToolDefinition",
    "UIContext",
    "NoopUIContext",
    "EventBus",
    # Infrastructure (Phase 7)
    "export_session_to_html",
    "TrustManager",
    "project_has_local_resources",
    "resolve_project_trusted",
    "process_at_files",
    "run_first_time_setup",
    "run_first_time_setup_sync",
    # Tools
    "create_all_tools",
    "create_coding_tools",
    "create_readonly_tools",
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_bash_tool",
    "create_grep_tool",
    "create_find_tool",
    "create_ls_tool",
]
