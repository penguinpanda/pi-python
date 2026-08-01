"""
pi-coding-agent  最小核心 CLI 编码代理

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
from ._session_manager import SessionManager
from ._cli import main
from ._print_mode import run_print_mode
from ._config import get_agent_dir, get_sessions_dir, load_settings
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
    # Modes
    "main",
    "run_print_mode",
    # Config
    "get_agent_dir",
    "get_sessions_dir",
    "load_settings",
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
