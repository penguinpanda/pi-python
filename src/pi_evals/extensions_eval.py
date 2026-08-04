"""extensions eval：临时扩展加载 + 命令注册 + prompt 端到端。"""

from __future__ import annotations

import pytest
from pi_agent import Agent, AgentOptions
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.extensions import ExtensionLoader, ExtensionRunner
from pi_coding_agent.model_runtime import ModelRuntime

from .harness import PiCodingAgentHarness


@pytest.mark.asyncio
async def test_extensions_eval(tmp_path):
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    core = faux_provider()
    core.set_responses([faux_assistant_message("with extension")])
    models.add_provider(core.provider)
    runtime = ModelRuntime(models, store)
    model = runtime.get_model("faux", "faux-1")
    assert model is not None

    extensions_dir = tmp_path / ".pi" / "extensions"
    extensions_dir.mkdir(parents=True)
    (extensions_dir / "sample.py").write_text(
        "def create_extension(api):\n"
        '    api.register_command("ext-hello", '
        '{"handler": lambda ctx, args: "hello from extension", '
        '"description": "Extension hello"})\n',
        encoding="utf-8",
    )
    loader = ExtensionLoader(
        global_dir=tmp_path / "nonexistent-global",
        project_dir=extensions_dir,
        cwd=str(tmp_path),
    )
    loaded = await loader.load()
    assert len(loaded.extensions) == 1
    runner = ExtensionRunner(
        loaded.extensions,
        runtime=loaded.runtime,
        cwd=str(tmp_path),
        model_runtime=runtime,
    )

    def factory() -> AgentSession:
        agent = Agent(
            AgentOptions(
                system_prompt="You are a helpful coding assistant.",
                model=model,
                stream_fn=runtime.stream,
            )
        )
        return AgentSession(
            agent=agent,
            session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
            cwd=str(tmp_path),
            model=model,
            model_runtime=runtime,
            extension_runner=runner,
        )

    harness = PiCodingAgentHarness(
        session_factory=factory,
        runtime=runtime,
        model={"provider": "faux", "id": "faux-1"},
    )
    result = await harness.run("run with the extension")
    assert result.errors == []
    assert result.output.strip() == "with extension"
    command_names = [command.name for command in runner.get_registered_commands()]
    assert "ext-hello" in command_names
