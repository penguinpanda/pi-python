"""Modal Editor - vim-like PiEditor subclass via set_editor_component.

Python port of modal-editor.ts（简化：Escape 切换 normal/insert，normal 下 hjkl 移动）。
"""

from pi_coding_agent.modes.interactive.components import PiEditor

from pi_coding_agent import ExtensionAPI


class ModalEditor(PiEditor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mode = "insert"
        self.border_title = " INSERT "

    def action_escape(self) -> None:
        if self._mode == "insert":
            self._mode = "normal"
            self.border_title = " NORMAL "
        else:
            self.post_message(self.Submitted(self, self.text))

    def action_i(self) -> None:
        self._mode = "insert"
        self.border_title = " INSERT "

    def action_h(self) -> None:
        if self._mode == "normal":
            self.action_cursor_left()

    def action_l(self) -> None:
        if self._mode == "normal":
            self.action_cursor_right()


def create_extension(pi: ExtensionAPI):
    def on_session_start(event, ctx):
        ctx.ui.set_editor_component(ModalEditor())

    pi.on("session_start", on_session_start)
