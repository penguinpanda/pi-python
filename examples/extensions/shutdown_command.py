"""Shutdown Command Extension - /quit and exit tools.

Python port of shutdown-command.ts.
"""

from pi_coding_agent import ExtensionAPI, ToolDefinition


def create_extension(pi: ExtensionAPI):
    pi.register_command(
        "quit",
        {
            "description": "Exit pi cleanly",
            "handler": lambda ctx, args: ctx.shutdown(),
        },
    )

    def finish_and_exit(tool_call_id, params, signal, on_update, ctx):
        ctx.shutdown()
        return {
            "content": [
                {"type": "text", "text": "Shutdown requested. Exiting after this response."}
            ],
            "details": {},
        }

    def deploy_and_exit(tool_call_id, params, signal, on_update, ctx):
        if on_update:
            on_update(
                {
                    "content": [
                        {"type": "text", "text": f"Deploying to {params['environment']}..."}
                    ],
                    "details": {},
                }
            )
        if on_update:
            on_update(
                {
                    "content": [{"type": "text", "text": "Deployment complete, exiting..."}],
                    "details": {},
                }
            )
        ctx.shutdown()
        return {
            "content": [{"type": "text", "text": "Done! Shutdown requested."}],
            "details": {"environment": params["environment"]},
        }

    pi.register_tool(
        ToolDefinition(
            name="finish_and_exit",
            label="Finish and Exit",
            description="Complete a task and exit pi",
            parameters={"type": "object", "properties": {}},
            execute=finish_and_exit,
        )
    )
    pi.register_tool(
        ToolDefinition(
            name="deploy_and_exit",
            label="Deploy and Exit",
            description="Deploy the application and exit pi",
            parameters={
                "type": "object",
                "properties": {
                    "environment": {
                        "type": "string",
                        "description": "Target environment (e.g., production, staging)",
                    }
                },
                "required": ["environment"],
            },
            execute=deploy_and_exit,
        )
    )
