"""resolve_tool_path 的 cwd 越界防护测试。"""

from __future__ import annotations

import os
import shutil
import uuid

import pytest

from pi_agent.env import FileError, PythonExecutionEnv
from pi_agent.tools.path_utils import is_path_within, resolve_tool_path


@pytest.fixture
def workdir() -> str:
    path = os.path.join(os.getcwd(), f"pi-test-guard-{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.asyncio
async def test_restricted_env_rejects_outside_path(workdir: str) -> None:
    env = PythonExecutionEnv(workdir, restrict_paths_to_cwd=True)
    resolved = await resolve_tool_path(env, "inside.txt")
    assert resolved == os.path.join(workdir, "inside.txt")

    with pytest.raises(FileError) as excinfo:
        await resolve_tool_path(env, "..")
    assert excinfo.value.code == "permission_denied"


@pytest.mark.asyncio
async def test_unrestricted_env_allows_outside_path(workdir: str) -> None:
    env = PythonExecutionEnv(workdir)  # 默认不限制
    resolved = await resolve_tool_path(env, "..")
    assert resolved == os.path.normpath(os.path.join(workdir, ".."))


def test_is_path_within() -> None:
    root = os.getcwd()
    assert is_path_within(os.path.join(root, "a", "b"), root)
    assert is_path_within(root, root)
    assert not is_path_within(os.path.dirname(root), root)
