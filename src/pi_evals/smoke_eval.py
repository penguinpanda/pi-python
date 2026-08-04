"""smoke eval：基本 prompt 端到端（faux provider，零网络）。"""

from __future__ import annotations

import pytest
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime

from .harness import PiCodingAgentHarness


def _runtime() -> ModelRuntime:
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    core = faux_provider()
    core.set_responses([faux_assistant_message("Paris")])
    models.add_provider(core.provider)
    return ModelRuntime(models, store)


@pytest.mark.asyncio
async def test_smoke_eval_basic_prompt():
    runtime = _runtime()
    harness = PiCodingAgentHarness(
        runtime=runtime,
        model={"provider": "faux", "id": "faux-1"},
    )
    result = await harness.run("What's the capital of France? Respond with only the city name.")
    assert result.output.strip() == "Paris"
    assert result.errors == []
    assert result.usage["provider"] == "faux"
    assert result.usage["model"] == "faux-1"
    assert result.usage["totalTokens"] > 0
