"""项目信任测试。"""

from __future__ import annotations

import json

import pytest

from pi_coding_agent.trust import (
    TrustManager,
    project_has_local_resources,
    resolve_project_trusted,
)


class TestTrustManager:
    def test_set_and_read(self, tmp_path):
        manager = TrustManager(tmp_path / "trust.json")
        manager.set_trust("/tmp/proj", True)
        assert manager.is_trusted("/tmp/proj") is True

    def test_ancestor_inheritance(self, tmp_path):
        manager = TrustManager(tmp_path / "trust.json")
        manager.set_trust("/tmp/root", True)
        assert manager.is_trusted("/tmp/root/sub/deep") is True

    def test_untrusted_ancestor_wins(self, tmp_path):
        manager = TrustManager(tmp_path / "trust.json")
        manager.set_trust("/tmp/root", True)
        manager.set_trust("/tmp/root/blocked", False)
        assert manager.is_trusted("/tmp/root/blocked/deep") is False

    def test_no_record_returns_none(self, tmp_path):
        manager = TrustManager(tmp_path / "trust.json")
        assert manager.is_trusted("/somewhere/else") is None

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "trust.json"
        TrustManager(path).set_trust("/tmp/p", True)
        assert TrustManager(path).is_trusted("/tmp/p") is True

    def test_reload_after_external_write(self, tmp_path):
        from pathlib import Path

        path = tmp_path / "trust.json"
        manager = TrustManager(path)
        canonical = str(Path("/tmp/x").resolve())
        path.write_text(json.dumps({canonical: False}), encoding="utf-8")
        manager.reload()
        assert manager.is_trusted("/tmp/x") is False


class TestProjectResources:
    def test_no_pi_dir(self, tmp_path):
        assert project_has_local_resources(str(tmp_path)) is False

    def test_with_skills_dir(self, tmp_path):
        (tmp_path / ".pi" / "skills").mkdir(parents=True)
        assert project_has_local_resources(str(tmp_path)) is True

    def test_only_settings_json(self, tmp_path):
        (tmp_path / ".pi").mkdir()
        (tmp_path / ".pi" / "settings.json").write_text("{}", encoding="utf-8")
        assert project_has_local_resources(str(tmp_path)) is False


class _FakeUI:
    def __init__(self, answer: bool) -> None:
        self.answer = answer

    async def confirm(self, title, message):
        return self.answer


class TestResolveProjectTrusted:
    async def test_override(self, tmp_path):
        (tmp_path / ".pi" / "skills").mkdir(parents=True)
        manager = TrustManager(tmp_path / "trust.json")
        assert (
            await resolve_project_trusted(
                str(tmp_path), manager, {"trustOverride": False}
            )
            is False
        )

    async def test_no_resources_auto_trust(self, tmp_path):
        manager = TrustManager(tmp_path / "trust.json")
        assert (
            await resolve_project_trusted(str(tmp_path), manager, {})
            is True
        )

    async def test_stored_decision(self, tmp_path):
        (tmp_path / ".pi" / "skills").mkdir(parents=True)
        manager = TrustManager(tmp_path / "trust.json")
        manager.set_trust(str(tmp_path), True)
        assert await resolve_project_trusted(str(tmp_path), manager, {}) is True

    async def test_default_trust(self, tmp_path):
        (tmp_path / ".pi" / "skills").mkdir(parents=True)
        manager = TrustManager(tmp_path / "trust.json")
        assert (
            await resolve_project_trusted(
                str(tmp_path), manager, {"defaultProjectTrust": "trust"}
            )
            is True
        )

    async def test_default_block(self, tmp_path):
        (tmp_path / ".pi" / "skills").mkdir(parents=True)
        manager = TrustManager(tmp_path / "trust.json")
        assert (
            await resolve_project_trusted(
                str(tmp_path), manager, {"defaultProjectTrust": "block"}
            )
            is False
        )

    async def test_ask_with_ui(self, tmp_path):
        (tmp_path / ".pi" / "skills").mkdir(parents=True)
        manager = TrustManager(tmp_path / "trust.json")
        assert (
            await resolve_project_trusted(
                str(tmp_path), manager, {}, ui=_FakeUI(True)
            )
            is True
        )
        # ask 后已持久化。
        assert manager.is_trusted(str(tmp_path)) is True

    async def test_ask_without_ui_denies(self, tmp_path):
        (tmp_path / ".pi" / "skills").mkdir(parents=True)
        manager = TrustManager(tmp_path / "trust.json")
        assert (
            await resolve_project_trusted(str(tmp_path), manager, {})
            is False
        )

    async def test_extension_decision(self, tmp_path):
        from pi_coding_agent.extensions.runner import ExtensionRunner
        from pi_coding_agent.extensions.types import Extension

        (tmp_path / ".pi" / "skills").mkdir(parents=True)
        extension = Extension(path="<inline>", resolved_path="<inline>")
        extension.handlers["project_trust"] = [
            lambda event, ctx: {"trusted": "no", "remember": True}
        ]
        runner = ExtensionRunner([extension], cwd=str(tmp_path))
        manager = TrustManager(tmp_path / "trust.json")
        assert (
            await resolve_project_trusted(
                str(tmp_path), manager, {}, extensions=runner
            )
            is False
        )
