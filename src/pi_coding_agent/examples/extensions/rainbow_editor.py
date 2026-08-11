"""Rainbow Editor - animated rainbow border when text contains 'ultrathink'.

Python port of rainbow-editor.ts（简化：用边框标题颜色动画代替逐字着色）。
"""

import asyncio

from pi_tui import PiEditor

from pi_coding_agent import ExtensionAPI


COLORS = ["#e98973", "#e4ba67", "#8dc07a", "#66c2b3", "#799dcf", "#9d86c3", "#ce82ac"]


class RainbowEditor(PiEditor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._frame = 0
        self._task: asyncio.Task | None = None
        self.border_title = " rainbow editor "

    async def _animate(self) -> None:
        while True:
            if "ultrathink" in self.text.lower():
                color = COLORS[self._frame % len(COLORS)]
                self.border_subtitle = f"ultrathink {color}"
                self._frame += 1
            await asyncio.sleep(0.06)

    def on_mount(self) -> None:
        super().on_mount()
        self._task = asyncio.create_task(self._animate())

    def on_unmount(self) -> None:
        if self._task is not None:
            self._task.cancel()
        super().on_unmount()


def create_extension(pi: ExtensionAPI):
    def on_session_start(event, ctx):
        ctx.ui.set_editor_component(RainbowEditor())

    pi.on("session_start", on_session_start)
