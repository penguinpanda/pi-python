"""Preset Extension - load presets.json and switch presets mid-session.

Python port of preset.ts（简化：/preset [name] 应用 model/thinking/tools；
instructions 提示用户用 --append-system-prompt）。
"""

import json
from pathlib import Path

from pi_coding_agent import ExtensionAPI


def _load_presets(ctx) -> dict:
    from pi_coding_agent._config import get_agent_dir

    presets: dict = {}
    for path in (
        get_agent_dir() / "presets.json",
        Path(ctx.cwd) / ".pi" / "presets.json",
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                presets.update(data)
        except (OSError, json.JSONDecodeError):
            continue
    return presets


def create_extension(pi: ExtensionAPI):
    async def handler(ctx, args: str) -> None:
        presets = _load_presets(ctx)
        if not args:
            if not presets:
                ctx.ui.notify("No presets found in presets.json", "warning")
                return
            name = await ctx.ui.select("Presets", list(presets))
            if name is None:
                return
        else:
            name = args.strip()
        preset = presets.get(name)
        if not isinstance(preset, dict):
            ctx.ui.notify(f"Preset not found: {name}", "error")
            return
        if isinstance(preset.get("provider"), str) and isinstance(preset.get("model"), str):
            from pi_ai import Model

            model = Model(
                id=preset["model"],
                provider=preset["provider"],
                api="openai-completions",
                name=preset["model"],
            )
            pi.set_model(model)
        if isinstance(preset.get("thinkingLevel"), str):
            pi.set_thinking_level(preset["thinkingLevel"])
        if isinstance(preset.get("tools"), list):
            pi.set_active_tools([str(tool) for tool in preset["tools"]])
        ctx.ui.notify(f"Applied preset: {name}", "info")

    pi.register_command(
        "preset",
        {
            "description": "Apply a named preset: /preset [name]",
            "handler": handler,
        },
    )
