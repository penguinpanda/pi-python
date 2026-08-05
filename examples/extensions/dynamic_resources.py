"""Dynamic Resources Extension - provide skills/prompts via resources_discover.

Python port of dynamic-resources/index.ts. 返回路径形式（skillPaths /
promptPaths / themePaths），runner 会加载为 Skill / PromptTemplate / Theme。
"""

from pathlib import Path

from pi_coding_agent import ExtensionAPI


BASE_DIR = Path(__file__).resolve().parent / "dynamic-resources"


def create_extension(pi: ExtensionAPI):
    def on_resources_discover(event, ctx):
        return {
            "skillPaths": [str(BASE_DIR / "SKILL.md")],
            "promptPaths": [str(BASE_DIR / "dynamic.md")],
        }

    pi.on("resources_discover", on_resources_discover)
