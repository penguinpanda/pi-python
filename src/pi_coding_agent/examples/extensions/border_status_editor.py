"""Border Status Editor - editor border shows model/thinking/context.

Python port of border-status-editor.ts（简化：用 border_title/subtitle 显示状态）。
"""

from pi_tui import PiEditor

from pi_coding_agent import ExtensionAPI


class BorderStatusEditor(PiEditor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model_text = "no model"
        self._thinking = "off"

    def update_status(self, model_text: str, thinking: str) -> None:
        self._model_text = model_text
        self._thinking = thinking
        self.border_title = f" {model_text} · thinking:{thinking} "


def create_extension(pi: ExtensionAPI):
    def on_session_start(event, ctx):
        editor = BorderStatusEditor()
        ctx.ui.set_editor_component(editor)
        model_text = f"{ctx.model.provider}/{ctx.model.id}" if ctx.model is not None else "no model"
        editor.update_status(model_text, ctx.thinking_level)

    pi.on("session_start", on_session_start)
