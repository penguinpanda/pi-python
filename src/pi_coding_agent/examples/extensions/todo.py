"""Todo Extension - state management via tool result details.

Python port of todo.ts。状态存于工具结果 details（随会话分支正确恢复）。
"""

from pi_coding_agent import ExtensionAPI, ToolDefinition


def create_extension(pi: ExtensionAPI):
    todos: list[dict] = []
    next_id = 1

    def reconstruct(ctx) -> None:
        nonlocal todos, next_id
        todos = []
        next_id = 1
        if ctx.session is None:
            return
        for entry in ctx.session.session_manager.get_branch():
            if entry.get("type") != "message":
                continue
            message = entry.get("message") or {}
            if message.get("role") != "toolResult" or message.get("tool_name") != "todo":
                continue
            details = message.get("details")
            if isinstance(details, dict):
                todos = list(details.get("todos") or [])
                next_id = int(details.get("nextId", 1))

    def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        nonlocal todos, next_id
        action = params.get("action", "list")
        if action == "add":
            text = params.get("text", "").strip()
            if not text:
                return {
                    "content": [{"type": "text", "text": "Missing text for add"}],
                    "details": {"action": action, "todos": todos, "nextId": next_id},
                }
            todos.append({"id": next_id, "text": text, "done": False})
            next_id += 1
            result = f"Added todo #{todos[-1]['id']}"
        elif action == "toggle":
            target = params.get("id")
            for todo in todos:
                if todo["id"] == target:
                    todo["done"] = not todo["done"]
                    result = f"Toggled todo #{target}"
                    break
            else:
                result = f"Todo #{target} not found"
        elif action == "clear":
            todos = [todo for todo in todos if not todo["done"]]
            result = "Cleared completed todos"
        else:
            result = (
                "No todos yet."
                if not todos
                else "\n".join(
                    f"{'[x]' if todo['done'] else '[ ]'} #{todo['id']} {todo['text']}"
                    for todo in todos
                )
            )
        return {
            "content": [{"type": "text", "text": result}],
            "details": {"action": action, "todos": todos, "nextId": next_id},
        }

    def todos_command(ctx, args: str) -> None:
        reconstruct(ctx)
        if not todos:
            ctx.ui.notify("No todos yet. Ask the agent to add some!", "info")
            return
        lines = [
            f"{sum(1 for t in todos if t['done'])}/{len(todos)} completed",
            *[f"{'[x]' if todo['done'] else '[ ]'} #{todo['id']} {todo['text']}" for todo in todos],
        ]
        ctx.ui.notify("\n".join(lines), "info")

    pi.register_tool(
        ToolDefinition(
            name="todo",
            label="Todo",
            description="Manage a todo list. Actions: list, add (text), toggle (id), clear",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "add", "toggle", "clear"],
                        "description": "Action to perform",
                    },
                    "text": {"type": "string", "description": "Todo text (for add)"},
                    "id": {"type": "number", "description": "Todo ID (for toggle)"},
                },
                "required": ["action"],
            },
            execute=execute,
        )
    )
    pi.register_command(
        "todos",
        {
            "description": "View the todo list",
            "handler": todos_command,
        },
    )

    def on_session_start(event, ctx):
        reconstruct(ctx)

    pi.on("session_start", on_session_start)
