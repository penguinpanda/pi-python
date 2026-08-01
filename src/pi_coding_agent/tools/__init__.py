"""内置编码工具集合。

提供 7 个编码工具: read, write, edit, bash, grep, find, ls
以及 3 种组合器: create_all_tools, create_coding_tools, create_readonly_tools
"""

from ._read import create_read_tool
from ._write import create_write_tool
from ._edit import create_edit_tool
from ._bash import create_bash_tool
from ._grep import create_grep_tool
from ._find import create_find_tool
from ._ls import create_ls_tool

__all__ = [
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


def create_all_tools(cwd: str) -> list:
    """全部 7 个工具。"""
    return [
        create_read_tool(cwd),
        create_write_tool(cwd),
        create_edit_tool(cwd),
        create_bash_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]


def create_coding_tools(cwd: str) -> list:
    """编码模式: read + bash + edit + write。"""
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


def create_readonly_tools(cwd: str) -> list:
    """只读探索: read + grep + find + ls。"""
    return [
        create_read_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]
