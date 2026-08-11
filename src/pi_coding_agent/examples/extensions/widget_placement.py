"""Widget Placement Extension - widgets above / below the editor.

Python port of widget-placement.ts。
"""

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    def on_session_start(event, ctx):
        if not ctx.has_ui:
            return
        ctx.ui.set_widget("widget-above", ["Above editor widget"])
        ctx.ui.set_widget(
            "widget-below",
            ["Below editor widget"],
            {"placement": "belowEditor"},
        )

    pi.on("session_start", on_session_start)
