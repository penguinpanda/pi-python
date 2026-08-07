"""extensions eval：扩展编写 + reload + 工具使用（对齐 TS extensions.eval.ts）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .harness import create_pi_coding_agent_harness
from .vitest_evals.harness_table import eval_harness_table
from .vitest_evals.judge import JudgeContext, create_judge
from .vitest_evals.suite import describe_eval

EXTENSION_NAME = "hello.py"
EXTENSION_CONTENT_TYPE = "text/x-python"


def _extension_errors(session: Any) -> list[dict[str, str]]:
    return list(getattr(session, "_pi_eval_extension_errors", []) or [])


def _require_section(default_prompt: str, marker: str) -> str:
    index = default_prompt.find(marker)
    if index == -1:
        raise RuntimeError(f"Default Pi system prompt has no {marker.strip()} section.")
    return default_prompt[:index]


def exclude_guidelines_and_documentation(default_prompt: str) -> str:
    """baseline：去掉 Guidelines 与 Pi 文档段（对齐 TS excludeGuidelinesAndDocumentation）。"""
    return _require_section(default_prompt, "\nGuidelines:\n")


def prepare_default_prompt_override(default_prompt: str) -> str:
    """candidate：只去掉 cwd 段（对齐 TS prepareDefaultPromptOverride）。"""
    return _require_section(default_prompt, "\nCurrent working directory: ")


def create_extension_authoring_harness(
    name: str,
    transform_system_prompt: Any = None,
):
    """创建扩展编写 harness（对齐 TS createExtensionAuthoringHarness）。"""

    def output(args: dict[str, Any]):
        session = args["session"]
        extension_path = Path(session.cwd) / ".pi" / "extensions" / EXTENSION_NAME
        extension_source = (
            extension_path.read_text(encoding="utf-8") if extension_path.exists() else None
        )
        runner = session.extension_runner
        loaded_extensions: list[dict[str, Any]] = []
        if runner is not None:
            loaded_extensions = [
                {"path": extension.path, "tools": list(extension.tools)}
                for extension in runner.extensions
            ]
        system_prompt = session._agent.state.system_prompt
        return {
            "response": args["response"],
            "systemPromptHasGuidelines": "\nGuidelines:\n" in system_prompt,
            "systemPromptHasPiDocs": "\nPi documentation (read only" in system_prompt,
            "extensionErrors": _extension_errors(session),
            "loadedExtensions": loaded_extensions,
            "extensionSource": extension_source,
        }

    return create_pi_coding_agent_harness(
        name=name,
        transform_system_prompt=transform_system_prompt,
        output=output,
    )


def _extension_authoring_judge(ctx: JudgeContext) -> dict[str, Any]:
    failures: list[str] = []
    output = ctx.output
    if not isinstance(output, dict):
        failures.append("output is unavailable")
    else:
        extension_source = output.get("extensionSource")
        if extension_source is None:
            failures.append("generated extension source is unavailable")
        elif isinstance(extension_source, str):
            imports = re.findall(r"(?:from|import)\s+([\w.]+)", extension_source)
            if "pi_coding_agent" not in imports:
                failures.append("extension does not import the canonical pi_coding_agent package")
            if any(specifier.startswith("pi_evals") for specifier in imports):
                failures.append("extension imports an eval-only package")
            if any(specifier.startswith("pi_tests") for specifier in imports):
                failures.append("extension imports a test-only package")
            if any(specifier.startswith("tests.") for specifier in imports):
                failures.append("extension imports a test package via tests. prefix")
        if output.get("extensionErrors"):
            failures.append("extension loader reported errors")
        loaded = output.get("loadedExtensions")
        has_hello_tool = False
        if isinstance(loaded, list):
            for extension in loaded:
                if not isinstance(extension, dict):
                    continue
                tools = extension.get("tools")
                if isinstance(tools, list) and "hello" in tools:
                    has_hello_tool = True
                    break
        if not has_hello_tool:
            failures.append('no loaded extension registered the "hello" tool')
        if not any(
            call.get("name") == "hello"
            and call.get("status") == "ok"
            and isinstance(call.get("arguments"), dict)
            and call["arguments"].get("name") == "Bob"
            and call.get("result") == "Hello, Bob!"
            for call in ctx.tool_calls
        ):
            failures.append('no successful hello({"name": "Bob"}) call returned "Hello, Bob!"')
        if output.get("response") != "Hello, Bob!":
            failures.append('final response was not exactly "Hello, Bob!"')
    return {
        "score": 1 if not failures else 0,
        "metadata": {
            "rationale": "Extension authoring workflow completed."
            if not failures
            else "; ".join(failures)
        },
    }


extension_authoring_judge = create_judge("ExtensionAuthoringJudge", _extension_authoring_judge)

extension_harness_table = eval_harness_table(
    "Pi extension authoring system prompt",
    baseline=create_extension_authoring_harness(
        "system-prompt-without-docs",
        exclude_guidelines_and_documentation,
    ),
    candidate=create_extension_authoring_harness(
        "default-system-prompt",
        prepare_default_prompt_override,
    ),
)

for _row in extension_harness_table:

    @describe_eval(
        f"{_row.name} repetition {_row.repetition}",
        harness=_row.harness,
        judges=[extension_authoring_judge],
        judge_threshold=None,
    )
    async def _extension_authoring_case(ctx, _row=_row):
        result = await ctx.run(
            [
                {
                    "type": "prompt",
                    "content": (
                        "Create a Pi extension with a hello tool that takes a name and "
                        "returns a greeting. For example, passing Bob should return "
                        "`Hello, Bob!`."
                    ),
                },
                {"type": "reload"},
                {
                    "type": "prompt",
                    "content": "Use the hello tool to greet Bob. Respond with exactly the "
                    "tool's greeting and nothing else.",
                },
            ]
        )
        output = result.output
        assert isinstance(output, dict)
        extension_source = output.get("extensionSource")
        if isinstance(extension_source, str):
            ctx.record_source_artifact(
                EXTENSION_NAME,
                EXTENSION_CONTENT_TYPE,
                extension_source,
            )
        expects_full_prompt = ctx.case.harness.name == "default-system-prompt"
        assert output.get("systemPromptHasGuidelines") is expects_full_prompt
        assert output.get("systemPromptHasPiDocs") is expects_full_prompt
