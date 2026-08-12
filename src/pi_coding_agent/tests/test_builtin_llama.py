"""llama.cpp 内置扩展测试。"""

from __future__ import annotations

from pi_coding_agent.extensions import Extension, ExtensionAPI, ExtensionRuntime
from pi_coding_agent.extensions.builtin_llama import LLAMA_PROVIDER_ID, create_extension


def test_builtin_llama_registers_provider_and_command() -> None:
    runtime = ExtensionRuntime()
    extension = Extension(
        path="<builtin>/llama",
        resolved_path="<builtin>/llama",
        source="builtin",
        hidden=True,
    )
    create_extension(ExtensionAPI(extension, runtime, cwd="."))

    assert extension.providers[0][0] == LLAMA_PROVIDER_ID
    provider_config = extension.providers[0][1]
    assert provider_config["api"] == "openai-completions"
    assert provider_config["models"][0]["id"] == "llama3"
    assert "llama" in extension.commands
