"""Plan Mode - read-only exploration with plan/todo tracking.

Python port of TS plan-mode/（完整对齐）：/plan、/todos、Ctrl+Alt+P、
--plan flag、bash allowlist、Plan 提取、[DONE:n] 进度、会话恢复。
"""

from __future__ import annotations

from pi_coding_agent import ExtensionAPI

from plan_mode_utils import extract_todo_items, is_safe_command, mark_completed_steps


PLAN_MODE_TOOLS = ["read", "bash", "grep", "find", "ls", "questionnaire"]
NORMAL_MODE_TOOLS = ["read", "bash", "edit", "write"]
PLAN_MODE_DISABLED_TOOLS = {"edit", "write"}
PLAN_MANAGED_TOOLS = set(PLAN_MODE_TOOLS) | set(NORMAL_MODE_TOOLS)

PLAN_CONTEXT_TEXT = """[PLAN MODE ACTIVE]
You are in plan mode - a read-only exploration mode for safe code analysis.

Restrictions:
- Built-in edit and write tools are disabled
- Other currently active tools remain available
- Bash is restricted to an allowlist of read-only commands

Ask clarifying questions using the questionnaire tool.
Use brave-search skill via bash for web research.

Create a detailed numbered plan under a "Plan:" header:

Plan:
1. First step description
2. Second step description
...

Do NOT attempt to make changes - just describe what you would do."""

EXECUTION_CONTEXT_TEXT = """[EXECUTING PLAN - Full tool access enabled]

Remaining steps:
{todos}

Execute each step in order.
After completing a step, include a [DONE:n] tag in your response."""


def _assistant_text(message) -> str:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ""
    parts: list[str] = []
    for block in message.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def create_extension(pi: ExtensionAPI):
    state = {
        "enabled": False,
        "executing": False,
        "todos": [],
        "tools_before": None,
    }

    def update_status(ctx) -> None:
        todos = state["todos"]
        if state["executing"] and todos:
            completed = sum(1 for item in todos if item["completed"])
            ctx.ui.set_status(
                "plan-mode",
                ctx.ui.theme.fg("accent", f"📋 {completed}/{len(todos)}"),
            )
        elif state["enabled"]:
            ctx.ui.set_status("plan-mode", ctx.ui.theme.fg("warning", "⏸ plan"))
        else:
            ctx.ui.set_status("plan-mode", None)

        if state["executing"] and todos:
            lines = []
            for item in todos:
                if item["completed"]:
                    lines.append(
                        ctx.ui.theme.fg("success", "☑ ")
                        + ctx.ui.theme.fg("textAlt", ctx.ui.theme.strikethrough(item["text"]))
                    )
                else:
                    lines.append(f"{ctx.ui.theme.fg('textAlt', '☐ ')}{item['text']}")
            ctx.ui.set_widget("plan-todos", lines)
        else:
            ctx.ui.set_widget("plan-todos", None)

    def unique_tool_names(names: list[str]) -> list[str]:
        return list(dict.fromkeys(names))

    def get_plan_mode_tools(active: list[str]) -> list[str]:
        return unique_tool_names(
            [name for name in active if name not in PLAN_MODE_DISABLED_TOOLS] + PLAN_MODE_TOOLS
        )

    def get_normal_mode_tools(active: list[str]) -> list[str]:
        return unique_tool_names(
            NORMAL_MODE_TOOLS + [name for name in active if name not in PLAN_MANAGED_TOOLS]
        )

    def persist_state() -> None:
        pi.append_entry(
            "plan-mode",
            {
                "enabled": state["enabled"],
                "todos": state["todos"],
                "executing": state["executing"],
                "tools_before": state["tools_before"],
            },
        )

    def toggle_plan_mode(ctx) -> None:
        state["enabled"] = not state["enabled"]
        state["executing"] = False
        state["todos"] = []
        if state["enabled"]:
            if state["tools_before"] is None:
                state["tools_before"] = pi.get_active_tools()
            pi.set_active_tools(get_plan_mode_tools(state["tools_before"]))
            ctx.ui.notify("Plan mode enabled. Built-in write tools disabled.")
        else:
            pi.set_active_tools(
                state["tools_before"] or get_normal_mode_tools(pi.get_active_tools())
            )
            state["tools_before"] = None
            ctx.ui.notify("Plan mode disabled. Full access restored.")
        update_status(ctx)
        persist_state()

    def restore_normal_mode_tools() -> None:
        pi.set_active_tools(state["tools_before"] or get_normal_mode_tools(pi.get_active_tools()))
        state["tools_before"] = None

    pi.register_flag(
        "plan",
        {
            "description": "Start in plan mode (read-only exploration)",
            "type": "boolean",
            "default": False,
        },
    )

    def on_session_start(event, ctx) -> None:
        if pi.get_flag("plan") is True:
            state["enabled"] = True
        if ctx.session_manager is not None:
            entries = ctx.session_manager.get_entries()
            plan_entry = None
            for entry in reversed(entries):
                if entry.get("type") == "custom" and entry.get("customType") == "plan-mode":
                    plan_entry = entry
                    break
            if plan_entry is not None and isinstance(plan_entry.get("data"), dict):
                data = plan_entry["data"]
                state["enabled"] = bool(data.get("enabled", state["enabled"]))
                state["todos"] = list(data.get("todos") or state["todos"])
                state["executing"] = bool(data.get("executing", state["executing"]))
                state["tools_before"] = data.get("tools_before", state["tools_before"])
            if state["executing"] and state["todos"]:
                execute_index = -1
                for index in range(len(entries) - 1, -1, -1):
                    if entries[index].get("customType") == "plan-mode-execute":
                        execute_index = index
                        break
                texts: list[str] = []
                for entry in entries[execute_index + 1 :]:
                    if entry.get("type") == "message" and isinstance(entry.get("message"), dict):
                        texts.append(_assistant_text(entry["message"]))
                mark_completed_steps("\n".join(texts), state["todos"])
        if state["enabled"]:
            if state["tools_before"] is None:
                state["tools_before"] = pi.get_active_tools()
            pi.set_active_tools(get_plan_mode_tools(state["tools_before"]))
        update_status(ctx)

    def on_tool_call(event, ctx):
        if not state["enabled"] or event.get("toolName") != "bash":
            return None
        command = str((event.get("input") or {}).get("command", ""))
        if not is_safe_command(command):
            return {
                "block": True,
                "reason": (
                    "Plan mode: command blocked (not allowlisted). "
                    "Use /plan to disable plan mode first.\nCommand: " + command
                ),
            }
        return None

    def on_context(event, ctx):
        if state["enabled"]:
            return None

        def _keep(message) -> bool:
            if message.get("customType") == "plan-mode-context":
                return False
            if message.get("role") != "user":
                return True
            content = message.get("content")
            if isinstance(content, str):
                return "[PLAN MODE ACTIVE]" not in content
            if isinstance(content, list):
                return not any(
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and "[PLAN MODE ACTIVE]" in str(block.get("text", ""))
                    for block in content
                )
            return True

        return {
            "messages": [message for message in (event.get("messages") or []) if _keep(message)]
        }

    def on_before_agent_start(event, ctx):
        if state["enabled"]:
            return {
                "message": {
                    "customType": "plan-mode-context",
                    "content": PLAN_CONTEXT_TEXT,
                    "display": False,
                }
            }
        if state["executing"] and state["todos"]:
            remaining = [item for item in state["todos"] if not item["completed"]]
            todo_list = "\n".join(f"{item['step']}. {item['text']}" for item in remaining)
            return {
                "message": {
                    "customType": "plan-execution-context",
                    "content": EXECUTION_CONTEXT_TEXT.format(todos=todo_list),
                    "display": False,
                }
            }
        return None

    def on_turn_end(event, ctx) -> None:
        if not state["executing"] or not state["todos"]:
            return
        text = _assistant_text(event.get("message"))
        if mark_completed_steps(text, state["todos"]) > 0:
            update_status(ctx)
        persist_state()

    async def on_agent_end(event, ctx) -> None:
        if state["executing"] and state["todos"]:
            if all(item["completed"] for item in state["todos"]):
                completed_list = "\n".join(f"~~{item['text']}~~" for item in state["todos"])
                pi.send_message(
                    {
                        "customType": "plan-complete",
                        "content": f"**Plan Complete!** ✓\n\n{completed_list}",
                        "display": True,
                    }
                )
                state["executing"] = False
                state["todos"] = []
                update_status(ctx)
                persist_state()
            return
        if not state["enabled"] or not ctx.has_ui:
            return
        last_assistant = None
        for message in reversed(event.get("messages") or []):
            if isinstance(message, dict) and message.get("role") == "assistant":
                last_assistant = message
                break
        if last_assistant is not None:
            extracted = extract_todo_items(_assistant_text(last_assistant))
            if extracted:
                state["todos"] = extracted
        if not state["todos"]:
            return
        persist_state()
        todo_list_text = "\n".join(
            f"{index + 1}. ☐ {item['text']}" for index, item in enumerate(state["todos"])
        )
        plan_todo_list_message = {
            "customType": "plan-todo-list",
            "content": f"**Plan Steps ({len(state['todos'])}):**\n\n{todo_list_text}",
            "display": True,
        }
        choice = await ctx.ui.select(
            "Plan mode - what next?",
            ["Execute the plan (track progress)", "Stay in plan mode", "Refine the plan"],
        )
        if choice and choice.startswith("Execute"):
            first = state["todos"][0]
            state["enabled"] = False
            state["executing"] = True
            restore_normal_mode_tools()
            update_status(ctx)
            persist_state()
            remaining_list = "\n".join(f"{item['step']}. {item['text']}" for item in state["todos"])
            exec_message = (
                "Execute the plan.\n\nRemaining steps:\n"
                f"{remaining_list}\n\nStart with: {first['text']}\n"
                "After completing a step, include a [DONE:n] tag in your response."
            )
            pi.send_message(
                plan_todo_list_message,
                {"customType": "plan-todo-list", "deliverAs": "followUp"},
            )
            pi.send_message(
                {
                    "customType": "plan-mode-execute",
                    "content": exec_message,
                    "display": True,
                },
                {
                    "customType": "plan-mode-execute",
                    "deliverAs": "followUp",
                    "triggerTurn": True,
                },
            )
        elif choice == "Refine the plan":
            refinement = await ctx.ui.editor("Refine the plan:", "")
            if refinement and refinement.strip():
                pi.send_message(
                    plan_todo_list_message,
                    {"customType": "plan-todo-list", "deliverAs": "followUp"},
                )
                pi.send_user_message(refinement.strip(), {"deliverAs": "followUp"})

    def plan_handler(ctx, args: str) -> None:
        toggle_plan_mode(ctx)

    def todos_handler(ctx, args: str) -> None:
        if not state["todos"]:
            ctx.ui.notify("No todos. Create a plan first with /plan", "info")
            return
        lines = "\n".join(
            f"{index + 1}. {'✓' if item['completed'] else '○'} {item['text']}"
            for index, item in enumerate(state["todos"])
        )
        ctx.ui.notify(f"Plan Progress:\n{lines}", "info")

    pi.on("session_start", on_session_start)
    pi.on("tool_call", on_tool_call)
    pi.on("context", on_context)
    pi.on("before_agent_start", on_before_agent_start)
    pi.on("turn_end", on_turn_end)
    pi.on("agent_end", on_agent_end)
    pi.register_command(
        "plan", {"description": "Toggle plan mode (read-only exploration)", "handler": plan_handler}
    )
    pi.register_command(
        "todos", {"description": "Show current plan todo list", "handler": todos_handler}
    )
    pi.register_shortcut(
        "ctrl+alt+p",
        {"description": "Toggle plan mode", "handler": lambda ctx: toggle_plan_mode(ctx)},
    )
