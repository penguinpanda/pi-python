"""Question Tool - ask the user with options via ctx.ui.

Python port of question.ts（简化：select + input，无自定义 Editor UI）。
"""

from pi_coding_agent import ExtensionAPI, ToolDefinition


def create_extension(pi: ExtensionAPI):
    async def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        question = str(params.get("question", ""))
        options = [str(option.get("label", option)) for option in params.get("options", [])]
        if ctx is None or not ctx.has_ui:
            return {
                "content": [{"type": "text", "text": "Error: UI not available"}],
                "details": {"question": question, "options": options, "answer": None},
            }
        if not options:
            return {
                "content": [{"type": "text", "text": "Error: No options provided"}],
                "details": {"question": question, "options": [], "answer": None},
            }
        choice = await ctx.ui.select(question, [*options, "Type something."])
        if choice is None:
            answer = None
        elif choice == "Type something.":
            answer = await ctx.ui.input(question, "Your answer")
        else:
            answer = choice
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"User answered: {answer}" if answer is not None else "No answer",
                }
            ],
            "details": {"question": question, "options": options, "answer": answer},
        }

    pi.register_tool(
        ToolDefinition(
            name="question",
            label="Question",
            description="Ask the user a question and let them pick from options",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask"},
                    "options": {
                        "type": "array",
                        "items": {"type": "object", "properties": {"label": {"type": "string"}}},
                        "description": "Options for the user to choose from",
                    },
                },
                "required": ["question", "options"],
            },
            execute=execute,
        )
    )
