"""ExtensionRunner / ExtensionRegistry 单元测试。"""

from __future__ import annotations

import pytest
from pi_ai import Models
from pi_ai.providers.faux import faux_provider

from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.extensions.registry import ExtensionRegistry
from pi_coding_agent.extensions.runner import ExtensionRunner
from pi_coding_agent.extensions.types import (
    Extension,
    ExtensionError,
)
from pi_coding_agent.model_runtime import ModelRuntime


def _make_extension(handlers: dict | None = None) -> Extension:
    extension = Extension(path="<inline>", resolved_path="<inline>")
    for event_type, handler in (handlers or {}).items():
        if isinstance(handler, list):
            extension.handlers.setdefault(event_type, []).extend(handler)
        else:
            extension.handlers.setdefault(event_type, []).append(handler)
    return extension


def _make_runtime() -> ModelRuntime:
    models = Models(credentials=AuthStorage.in_memory())
    models.add_provider(faux_provider().provider)
    return ModelRuntime(models, AuthStorage.in_memory())


class TestEventDispatch:
    async def test_collects_handler_results_in_order(self):
        calls: list[str] = []

        def handler_one(event, ctx):
            calls.append("one")
            return 1

        def handler_two(event, ctx):
            calls.append("two")
            return 2

        runner = ExtensionRunner(
            [
                _make_extension({"agent_start": [handler_one]}),
                _make_extension({"agent_start": [handler_two]}),
            ]
        )
        results = await runner.emit_event("agent_start", {"prompt": "hi"})
        assert results == [1, 2]
        assert calls == ["one", "two"]

    async def test_handler_error_is_isolated(self):
        errors: list[ExtensionError] = []

        def bad(event, ctx):
            raise ValueError("boom")

        def good(event, ctx):
            return "ok"

        runner = ExtensionRunner(
            [
                _make_extension({"agent_start": [bad, good]}),
            ]
        )
        runner.on_error(errors.append)
        results = await runner.emit_event("agent_start")
        assert results == ["ok"]
        assert len(errors) == 1
        assert errors[0].event == "agent_start"
        assert "boom" in errors[0].error

    async def test_async_handlers(self):
        async def handler(event, ctx):
            return "async"

        runner = ExtensionRunner([_make_extension({"turn_start": [handler]})])
        assert await runner.emit_event("turn_start") == ["async"]

    def test_has_handlers(self):
        runner = ExtensionRunner([_make_extension({"input": [lambda e, c: None]})])
        assert runner.has_handlers("input") is True
        assert runner.has_handlers("agent_end") is False


class TestEmitInput:
    async def test_transform_chain(self):
        def transform(event, ctx):
            return {"action": "transform", "text": f"ext:{event['text']}"}

        runner = ExtensionRunner([_make_extension({"input": [transform]})])
        text, action = await runner.emit_input("hello")
        assert text == "ext:hello"
        assert action == "continue"

    async def test_handled_short_circuits(self):
        def handled(event, ctx):
            return {"action": "handled", "text": "stop"}

        def never(event, ctx):
            raise AssertionError("should not run")

        runner = ExtensionRunner(
            [
                _make_extension({"input": [handled]}),
                _make_extension({"input": [never]}),
            ]
        )
        text, action = await runner.emit_input("go")
        assert text == "stop"
        assert action == "handled"


class TestContext:
    def test_context_resolves_session_state(self):
        class FakeSession:
            model = object()
            thinking_level = "high"
            is_streaming = False
            pending_message_count = 0

            async def compact(self):
                return None

        runner = ExtensionRunner(cwd="/tmp/proj")
        runner.session = FakeSession()
        context = runner.create_context()
        assert context.cwd == "/tmp/proj"
        assert context.model is FakeSession.model
        assert context.thinking_level == "high"
        assert context.is_idle() is True
        assert context.has_pending_messages() is False

    async def test_command_context_new_session(self):
        calls: list[str] = []
        runner = ExtensionRunner()
        runner.bind(
            command_handlers={
                "new_session": lambda options: calls.append("new") or {"cancelled": False},
            }
        )
        context = runner.create_command_context()
        result = await context.new_session()
        assert result == {"cancelled": False}
        assert calls == ["new"]

    async def test_unknown_command_action_raises(self):
        runner = ExtensionRunner()
        context = runner.create_command_context()
        with pytest.raises(NotImplementedError):
            await context.fork("entry-1")


class TestRegistrations:
    def test_command_aggregation_with_duplicates(self):
        def make(command_name):
            extension = Extension(path="<inline>", resolved_path="<inline>")
            extension.commands[command_name] = type(
                "C",
                (),
                {
                    "name": command_name,
                    "description": "d",
                    "argument_hint": None,
                    "handler": None,
                    "source_info": None,
                },
            )()
            return extension

        runner = ExtensionRunner([make("cmd"), make("cmd"), make("other")])
        names = [command.name for command in runner.get_registered_commands()]
        assert names == ["cmd:1", "cmd:2", "other"]

    def test_flags_and_shortcuts(self):
        extension = Extension(path="<inline>", resolved_path="<inline>")
        extension.flags["f"] = type(
            "F",
            (),
            {
                "name": "f",
                "description": "",
                "type": "boolean",
                "default": True,
                "extension_path": "x",
            },
        )()
        extension.shortcuts["ctrl+k"] = type(
            "S",
            (),
            {"shortcut": "ctrl+k", "description": "s", "handler": None, "extension_path": "x"},
        )()
        runner = ExtensionRunner([extension])
        assert runner.get_flags()[0].name == "f"
        assert runner.get_shortcuts()[0].shortcut == "ctrl+k"

    async def test_provider_application(self):
        runtime = _make_runtime()
        extension = Extension(path="<inline>", resolved_path="<inline>")
        extension.providers.append(
            (
                "acme",
                {
                    "api_key": "sk-acme",
                    "base_url": "https://acme.api/v1",
                    "models": [{"id": "acme-1", "api": "openai-completions", "reasoning": False}],
                },
            )
        )
        runner = ExtensionRunner([extension], cwd="/tmp", model_runtime=runtime)
        runner.apply_providers()
        model = runtime.get_model("acme", "acme-1")
        assert model is not None
        assert model.base_url == "https://acme.api/v1"

    def test_registry_apply_commands_and_shortcuts(self):
        from pi_tui.keybindings import KeybindingsManager

        from pi_coding_agent.modes.interactive.slash_commands import SlashCommandRegistry

        extension = Extension(path="<inline>", resolved_path="<inline>")

        async def handler(ctx, args):
            return f"ran {args}"

        extension.commands["greet"] = type(
            "C",
            (),
            {
                "name": "greet",
                "description": "Greet",
                "argument_hint": "<name>",
                "handler": handler,
                "source_info": None,
            },
        )()
        extension.shortcuts["ctrl+k"] = type(
            "S",
            (),
            {
                "shortcut": "ctrl+k",
                "description": "Shortcut",
                "handler": None,
                "extension_path": "x",
            },
        )()

        runner = ExtensionRunner([extension], cwd="/tmp")
        slash_registry = SlashCommandRegistry()
        keybindings = KeybindingsManager()
        ExtensionRegistry(
            runner,
            slash_registry=slash_registry,
            keybindings_manager=keybindings,
        ).apply()

        assert slash_registry.get("greet") is not None
        assert keybindings.resolve("ctrl+k") is not None

    def test_registry_passthrough(self, tmp_path):
        extension = _make_extension()
        extension.commands["cmd"] = type(
            "C",
            (),
            {
                "name": "cmd",
                "description": "Cmd",
                "argument_hint": "<x>",
                "handler": lambda ctx, args: "x",
                "source_info": None,
            },
        )()
        extension.shortcuts["ctrl+j"] = type(
            "S",
            (),
            {"shortcut": "ctrl+j", "description": "s", "handler": None, "extension_path": "x"},
        )()
        extension.flags["flag"] = type(
            "F",
            (),
            {"name": "flag", "description": "f", "type": "boolean", "default": False},
        )()
        extension.providers.append(("acme", {"api_key": "sk"}))
        runner = ExtensionRunner([extension], cwd=str(tmp_path))
        registry = ExtensionRegistry(runner)
        assert [c.name for c in registry.get_commands()] == ["cmd"]
        assert [s.shortcut for s in registry.get_shortcuts()] == ["ctrl+j"]
        assert [f.name for f in registry.get_flags()] == ["flag"]
        assert registry.get_providers() == [("acme", {"api_key": "sk"})]
        assert registry.get_tools() == []


class TestExtensionApiRegistration:
    def test_register_shortcut_flag_provider_renderers(self, tmp_path):
        from pi_coding_agent.extensions.types import ExtensionAPI

        extension = Extension(path="<inline>", resolved_path="<inline>")
        runner = ExtensionRunner([extension], cwd=str(tmp_path))
        api = ExtensionAPI(extension, runner.runtime, cwd=str(tmp_path))

        api.register_shortcut("ctrl+l", {"description": "Clear"})
        api.register_flag("verbose", {"description": "Verbose", "type": "boolean"})
        api.register_provider("acme", {"apiKey": "$ACME_KEY", "baseUrl": "https://acme"})
        api.register_message_renderer("note", lambda *a: None)
        api.register_tool_renderer("bash", lambda *a: None)

        assert extension.shortcuts["ctrl+l"].description == "Clear"
        assert extension.flags["verbose"].description == "Verbose"
        assert extension.providers == [("acme", {"apiKey": "$ACME_KEY", "baseUrl": "https://acme"})]
        assert "note" in extension.message_renderers
        assert "bash" in extension.tool_renderers


class TestRunnerErrors:
    def test_emit_error_dispatches_and_unsubscribe(self):
        runner = ExtensionRunner([], cwd="/tmp")
        errors: list[ExtensionError] = []
        unsubscribe = runner.on_error(errors.append)
        runner.emit_error(ExtensionError("a.py", "agent_start", "boom", None))
        assert len(errors) == 1
        assert errors[0].event == "agent_start"
        unsubscribe()
        runner.emit_error(ExtensionError("a.py", "agent_end", "boom2", None))
        assert len(errors) == 1
