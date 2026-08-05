"""Reload Runtime Extension - ctx.reload() + LLM tool that queues /reload-runtime.

Python port of reload-runtime.ts。
"""

from pi_coding_agent import ExtensionAPI, ToolDefinition


def create_extension(pi: ExtensionAPI):
    async def reload_command(ctx, args: str) -> None:
        await ctx.reload()

    def reload_tool(tool_call_id, params, signal=None, on_update=None, ctx=None):
        pi.send_user_message("/reload-runtime", {"deliverAs": "followUp"})
        return {
            "content": [{"type": "text", "text": "Queued /reload-runtime as a follow-up command."}],
            "details": {},
        }

    pi.register_command(
        "reload-runtime",
        {
            "description": "Reload extensions, skills, prompts, themes, and context files",
            "handler": reload_command,
        },
    )
    pi.register_tool(
        ToolDefinition(
            name="reload_runtime",
            label="Reload Runtime",
            description="Reload extensions, skills, prompts, themes, and context files",
            parameters={"type": "object", "properties": {}},
            execute=reload_tool,
        )
    )
