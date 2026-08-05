"""Working Message Persistence Test - set_working_message.

Python port of working-message-test.ts（用 set_working_message + set_status）。
"""

from pi_coding_agent import ExtensionAPI


CUSTOM_MESSAGE = "\x1b[38;2;155;86;63mWorking... (custom)\x1b[39m"


def create_extension(pi: ExtensionAPI):
    def on_session_start(event, ctx):
        ctx.ui.set_working_message(CUSTOM_MESSAGE)
        ctx.ui.set_status("working-indicator", "\x1b[38;2;155;86;63m●\x1b[39m")

    pi.on("session_start", on_session_start)
