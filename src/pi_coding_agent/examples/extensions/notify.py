"""Pi Notify Extension - desktop notification when the agent is done.

Python port of notify.ts. Supports:
- OSC 777: Ghostty, iTerm2, WezTerm, rxvt-unicode
- OSC 99: Kitty
- Windows toast: Windows Terminal (via powershell)
"""

import os
import subprocess
import sys

from pi_coding_agent import ExtensionAPI


def _notify_osc777(title: str, body: str) -> None:
    sys.stdout.write(f"\x1b]777;notify;{title};{body}\x07")
    sys.stdout.flush()


def _notify_osc99(title: str, body: str) -> None:
    sys.stdout.write(f"\x1b]99;i=1:d=0;{title}\x1b\\")
    sys.stdout.write(f"\x1b]99;i=1:p=body;{body}\x1b\\")
    sys.stdout.flush()


def _notify_windows(title: str, body: str) -> None:
    ns = "Windows.UI.Notifications"
    mgr = f"[{ns}.ToastNotificationManager, {ns}, ContentType = WindowsRuntime]"
    template = f"[{ns}.ToastTemplateType]::ToastText01"
    toast = f"[{ns}.ToastNotification]::new($xml)"
    script = "; ".join(
        [
            f"{mgr} > $null",
            f"$xml = [{ns}.ToastNotificationManager]::GetTemplateContent({template})",
            "$xml.GetElementsByTagName('text')[0]."
            f"AppendChild($xml.CreateTextNode('{body}')) > $null",
            f"[{ns}.ToastNotificationManager]::CreateToastNotifier('{title}').Show({toast})",
        ]
    )
    subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-Command", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _notify(title: str, body: str) -> None:
    if os.environ.get("WT_SESSION"):
        _notify_windows(title, body)
    elif os.environ.get("KITTY_WINDOW_ID"):
        _notify_osc99(title, body)
    else:
        _notify_osc777(title, body)


def create_extension(pi: ExtensionAPI):
    async def on_agent_end(event, ctx):
        _notify("Pi", "Ready for input")

    pi.on("agent_end", on_agent_end)
