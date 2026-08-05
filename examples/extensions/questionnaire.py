"""Questionnaire Tool - ask multiple questions sequentially.

Python port of questionnaire.ts（简化：逐题 select/input，无 tab bar UI）。
"""

from pi_coding_agent import ExtensionAPI, ToolDefinition


def create_extension(pi: ExtensionAPI):
    async def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        questions = params.get("questions", [])
        answers: list[dict] = []
        if ctx is None or not ctx.has_ui:
            return {
                "content": [{"type": "text", "text": "Error: UI not available"}],
                "details": {"answers": answers, "cancelled": True},
            }
        for question in questions:
            prompt = str(question.get("prompt", ""))
            options = [str(option.get("label", option)) for option in question.get("options", [])]
            allow_other = bool(question.get("allowOther", True))
            choices = list(options)
            if allow_other:
                choices.append("Type something.")
            choice = await ctx.ui.select(prompt, choices)
            if choice is None:
                return {
                    "content": [{"type": "text", "text": "Cancelled"}],
                    "details": {"answers": answers, "cancelled": True},
                }
            value = choice
            was_custom = False
            if allow_other and choice == "Type something.":
                value = await ctx.ui.input(prompt, "Your answer")
                was_custom = True
                if value is None:
                    return {
                        "content": [{"type": "text", "text": "Cancelled"}],
                        "details": {"answers": answers, "cancelled": True},
                    }
            answers.append(
                {
                    "id": question.get("id", ""),
                    "label": choice,
                    "value": value,
                    "wasCustom": was_custom,
                }
            )
        return {
            "content": [{"type": "text", "text": "Answers collected"}],
            "details": {"answers": answers, "cancelled": False},
        }

    pi.register_tool(
        ToolDefinition(
            name="questionnaire",
            label="Questionnaire",
            description="Ask the user multiple questions sequentially",
            parameters={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "prompt": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {"label": {"type": "string"}},
                                    },
                                },
                                "allowOther": {"type": "boolean"},
                            },
                            "required": ["id", "prompt", "options"],
                        },
                    }
                },
                "required": ["questions"],
            },
            execute=execute,
        )
    )
