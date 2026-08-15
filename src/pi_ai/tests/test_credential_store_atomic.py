"""FileCredentialStore._save 原子写回归测试。

覆盖：

- stale tmp 清理 + 正常写入
- symlink 攻击：tmp 路径被替换为符号链接时不跟随写入
- (POSIX) 最终文件权限 0600

沙箱环境无 pytest tmp_path 且 tempfile.mkdtemp 目录受 ACL 限制，
改用仓库内 os.makedirs 临时目录并自清理。
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

import pytest

from pi_ai.auth.credential_store import FileCredentialStore


@pytest.fixture
def store_dir() -> str:
    path = os.path.join(os.getcwd(), f"pi-test-cred-{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _can_symlink(dirpath: str) -> bool:
    probe = Path(dirpath) / "probe-src"
    link = Path(dirpath) / "probe-link"
    try:
        probe.write_text("x", encoding="utf-8")
        os.symlink(probe, link)
        link.unlink()
        return True
    except (OSError, NotImplementedError):
        return False


@pytest.mark.asyncio
async def test_save_cleans_stale_tmp_and_writes(store_dir: str) -> None:
    path = Path(store_dir) / "auth.json"
    (Path(store_dir) / "auth.json.tmp").write_text("stale", encoding="utf-8")
    store = FileCredentialStore(path)
    await store.write("openai", {"type": "api_key", "key": "sk-1"})
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["openai"]["key"] == "sk-1"
    assert not (Path(store_dir) / "auth.json.tmp").exists()


@pytest.mark.asyncio
async def test_save_does_not_follow_symlink(store_dir: str) -> None:
    if not _can_symlink(store_dir):
        pytest.skip("no symlink privilege on this platform")
    path = Path(store_dir) / "auth.json"
    target = Path(store_dir) / "victim.txt"
    target.write_text("untouched", encoding="utf-8")
    os.symlink(target, Path(store_dir) / "auth.json.tmp")
    store = FileCredentialStore(path)
    await store.write("openai", {"type": "api_key", "key": "sk-2"})
    # 链接本身被移除并替换为真实文件；目标内容不变。
    assert target.read_text(encoding="utf-8") == "untouched"
    assert not os.path.islink(Path(store_dir) / "auth.json.tmp")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["openai"]["key"] == "sk-2"


@pytest.mark.skipif(os.name == "nt", reason="Windows chmod 不影响 ACL")
@pytest.mark.asyncio
async def test_save_final_mode_0600(store_dir: str) -> None:
    path = Path(store_dir) / "auth.json"
    store = FileCredentialStore(path)
    await store.write("openai", {"type": "api_key", "key": "sk-3"})
    assert (path.stat().st_mode & 0o777) == 0o600
