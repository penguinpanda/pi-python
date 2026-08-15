"""FileCredentialStore 原子写回归测试（atomic_write_json 语义）。

覆盖：

- 写入后目录无临时文件残留（随机后缀 + os.replace）
- symlink 攻击：目标路径被替换为符号链接时 os.replace 不跟随写入
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
from pi_ai.auth.credential_store import CredentialStoreCorruptError


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
async def test_save_leaves_no_tmp_leftovers(store_dir: str) -> None:
    path = Path(store_dir) / "auth.json"
    store = FileCredentialStore(path)
    await store.write("openai", {"type": "api_key", "key": "sk-1"})
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["openai"]["key"] == "sk-1"
    leftovers = [p for p in Path(store_dir).iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


@pytest.mark.asyncio
async def test_save_replace_does_not_follow_symlink(store_dir: str) -> None:
    if not _can_symlink(store_dir):
        pytest.skip("no symlink privilege on this platform")
    path = Path(store_dir) / "auth.json"
    target = Path(store_dir) / "victim.txt"
    target.write_text("untouched", encoding="utf-8")
    os.symlink(target, path)
    store = FileCredentialStore(path)
    # 写入前 _load 跟随 symlink 读到非 JSON：拒绝写入（备份 + 抛错），
    # 不会把凭证写进攻击者指定的目标文件。
    with pytest.raises(CredentialStoreCorruptError):
        await store.write("openai", {"type": "api_key", "key": "sk-2"})
    # 目标内容不变；被劫持的路径被备份而非覆盖。
    assert target.read_text(encoding="utf-8") == "untouched"
    assert not path.exists()
    assert (Path(store_dir) / "auth.json.corrupt").exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows chmod 不影响 ACL")
@pytest.mark.asyncio
async def test_save_final_mode_0600(store_dir: str) -> None:
    path = Path(store_dir) / "auth.json"
    store = FileCredentialStore(path)
    await store.write("openai", {"type": "api_key", "key": "sk-3"})
    assert (path.stat().st_mode & 0o777) == 0o600
