"""ExtensionRunner 补充测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pi_ai import Models
from pi_ai.providers.faux import faux_provider

from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.extensions.runner import ExtensionRunner
from pi_coding_agent.extensions.types import Extension, ExtensionError
from pi_coding_agent.model_runtime import ModelRuntime


def _make_extension(handlers: dict | None = None) -> Extension:
    extension = Extension(path="<inline>", resolved_path="<inline>")
    for event_type, handler in (handlers or {}).items():
        if isinstance(handler, list):
            extension.handlers.setdefault(event_type, []).extend(handler)
        else:
            extension.handlers.setdefault(event_type, []).append(handler)
    return extension


def _runtime() -> ModelRuntime:
    models = Models(credentials=AuthStorage.in_memory())
    models.add_provider(faux_provider().provider)
    return ModelRuntime(models, AuthStorage.in_memory())


class _FakeSession:
    def __init__(self) -> None:
        self.thinking_level = "high"
        self.session_name = "n"
        self.is_streaming = False
        self.pending_message_count = 0
        self.extension_state = {}
        self.follow_up_calls: list[str] = []
        self.steer_calls: list[str] = []
        self.prompt_calls: list[str] = []
        self.custom_messages: list[tuple] = []
        self.custom_entries: list[tuple] = []
        self.labels: list[tuple] = []
        self.model = object()
        self._extension_runner = None
        self._agent = SimpleNamespace(
            state=SimpleNamespace(
                tools=[],
            )
        )
        self._session_manager = SimpleNamespace(
            append_custom_message_entry=self._append_custom_message,
            append_custom_entry=self._append_custom_entry,
            set_label=lambda entry_id, label: self.labels.append((entry_id, label)),
        )

    async def _append_custom_message(self, *args, **kwargs):
        self.custom_messages.append((args, kwargs))
        return "entry-1"

    async def _append_custom_entry(self, *args, **kwargs):
        self.custom_entries.append((args, kwargs))
        return "entry-2"

    async def set_model(self, model):
        self.model = model
        return True

    def set_thinking_level(self, level):
        self.thinking_level = level

    def set_session_name(self, name):
        self.session_name = name

    def follow_up(self, text: str):
        self.follow_up_calls.append(text)

    def steer(self, text: str):
        self.steer_calls.append(text)

    async def prompt(self, text: str):
        self.prompt_calls.append(text)

    async def continue_(self):
        pass

    async def abort(self):
        pass

    def rebuild_system_prompt(self):
        pass


def test_bind_actions_and_handlers() -> None:
    runner = ExtensionRunner()
    runner.bind(
        ui_context=object(),
        mode="tui",
        shutdown_handler=lambda: None,
        abort_fn=lambda: None,
        actions={"custom": lambda: "ok"},
    )
    assert runner.mode == "tui"
    assert runner.runtime.get_action("custom")() == "ok"
    assert runner._shutdown_handler is not None
    assert runner._abort_fn is not None


def test_bind_session_registers_actions() -> None:
    session = _FakeSession()
    runner = ExtensionRunner()
    runner.bind_session(session)
    assert runner.runtime.get_action("get_thinking_level")() == "high"
    assert runner.runtime.get_action("get_session_name")() == "n"
    runner.runtime.get_action("set_session_name")("new")
    assert session.session_name == "new"


@pytest.mark.asyncio
async def test_run_send_message_followup_and_steer() -> None:
    session = _FakeSession()
    runner = ExtensionRunner()
    await runner._run_send_message(session, {"content": "hi"}, {"deliverAs": "followUp"})
    assert session.follow_up_calls == ["hi"]
    assert len(session.custom_messages) == 1

    await runner._run_send_message(session, "steer text", {"deliverAs": "steer"})
    assert session.steer_calls == ["steer text"]


@pytest.mark.asyncio
async def test_append_entry_and_set_label() -> None:
    session = _FakeSession()
    runner = ExtensionRunner()
    runner._action_append_entry(session, "note", {"x": 1})
    runner._action_set_label(session, "e1", "label")
    await asyncio.sleep(0.05)
    assert session.custom_entries
    assert session.labels == [("e1", "label")]


@pytest.mark.asyncio
async def test_emit_project_trust_paths() -> None:
    def yes(event, ctx):
        return {"trusted": "yes"}

    def no(event, ctx):
        return {"trusted": "no"}

    def undecided(event, ctx):
        return {"trusted": "undecided"}

    runner = ExtensionRunner([_make_extension({"project_trust": [yes]})])
    assert await runner.emit_project_trust("/tmp") == "yes"

    runner = ExtensionRunner([_make_extension({"project_trust": [undecided, no]})])
    assert await runner.emit_project_trust("/tmp") == "no"


@pytest.mark.asyncio
async def test_discover_resources_object_forms() -> None:
    def discover(event, ctx):
        return {
            "skills": [{"name": "s"}],
            "prompts": [{"name": "p"}],
            "themes": [{"name": "t"}],
        }

    runner = ExtensionRunner([_make_extension({"resources_discover": [discover]})])
    await runner.discover_resources()
    assert runner.get_discovered_skills() == [{"name": "s"}]
    assert runner.get_discovered_prompts() == [{"name": "p"}]
    assert runner.get_discovered_themes() == [{"name": "t"}]


def test_apply_providers_error_listener() -> None:
    runtime = _runtime()
    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.providers.append(("acme", {"models": [{"id": "x"}]}))
    runner = ExtensionRunner([extension], model_runtime=runtime)
    errors: list[ExtensionError] = []
    runner.on_error(errors.append)
    runner.apply_providers()
    assert errors
    assert errors[0].event == "register_provider"


def test_registration_aggregation_and_renderers() -> None:
    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.tools["read"] = SimpleNamespace(name="read")
    extension.message_renderers["note"] = lambda *a: None
    extension.tool_renderers["bash"] = lambda *a: None
    extension.entry_renderers["card"] = lambda *a: None
    extension.markdown_transformers.append(lambda *a: None)
    extension.autocomplete.append(lambda text: [])
    runner = ExtensionRunner([extension])
    assert runner.get_registered_tools()[0].name == "read"
    assert runner.get_tool_definition("read") is not None
    assert runner.get_tool_definition("missing") is None
    assert runner.get_message_renderer("note") is not None
    assert runner.get_tool_renderer("bash") is not None
    assert runner.get_entry_renderer("card") is not None
    assert len(runner.get_markdown_transformers()) == 1
    assert len(runner.get_autocomplete()) == 1


@pytest.mark.asyncio
async def test_active_tools_and_model_action() -> None:
    session = _FakeSession()
    runner = ExtensionRunner()
    assert runner._get_active_tools(None) == []
    assert runner._get_all_tools(None) == []
    assert await runner._action_set_model(None) is False

    tool = SimpleNamespace(
        name="read",
        description="d",
        input_schema={},
        prompt_guidelines="g",
    )
    session._agent.state.tools = [tool]
    assert runner._get_active_tools(session) == ["read"]
    assert runner._get_all_tools(session) == [
        {
            "name": "read",
            "description": "d",
            "parameters": {},
            "prompt_guidelines": "g",
            "source_info": {},
        }
    ]


@pytest.mark.asyncio
async def test_abort_shutdown_and_shutdown_all() -> None:
    calls: list[str] = []
    runner = ExtensionRunner(
        [_make_extension({"session_shutdown": [lambda e, c: calls.append("shutdown")]})]
    )
    runner.bind(
        abort_fn=lambda: calls.append("abort"), shutdown_handler=lambda: calls.append("exit")
    )
    runner.abort()
    runner.shutdown()
    await runner.shutdown_all()
    assert calls == ["abort", "exit", "shutdown"]
