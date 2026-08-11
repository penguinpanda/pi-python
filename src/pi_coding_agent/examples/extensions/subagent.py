"""Subagent Extension - delegate a task to an isolated in-process agent.

Python port of subagent/（简化：新建独立 AgentSession 跑单轮，结果写入编辑器）。
"""

from pi_coding_agent import ExtensionAPI


SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused subagent with an isolated context window. "
    "Complete the given task, then summarize what you did."
)


def create_extension(pi: ExtensionAPI):
    async def handler(ctx, args: str) -> None:
        goal = args.strip()
        if not goal:
            ctx.ui.notify("Usage: /subagent <goal>", "warning")
            return
        if ctx.model is None or ctx.session is None:
            ctx.ui.notify("No model selected", "error")
            return
        from pi_agent import Agent, AgentOptions

        from pi_coding_agent import AgentSession
        from pi_coding_agent._session_manager import SessionManager

        stream_fn = ctx.session._agent.stream_function
        agent = Agent(
            AgentOptions(
                system_prompt=SUBAGENT_SYSTEM_PROMPT,
                model=ctx.model,
                stream_fn=stream_fn,
            )
        )
        manager = SessionManager.in_memory(cwd=ctx.cwd)
        subagent = AgentSession(
            agent=agent,
            session_manager=manager,
            cwd=ctx.cwd,
            model=ctx.model,
        )
        try:
            await subagent.prompt(goal)
            result = subagent.get_last_assistant_text() or "(no output)"
        finally:
            await subagent.dispose()
        ctx.ui.set_editor_text(result)
        ctx.ui.notify("Subagent finished", "info")

    pi.register_command(
        "subagent",
        {
            "description": "Delegate a task to an isolated subagent",
            "handler": handler,
        },
    )
