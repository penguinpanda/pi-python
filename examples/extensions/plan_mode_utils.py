"""Plan Mode 纯工具函数（对齐 TS plan-mode/utils.ts）。"""

from __future__ import annotations

import re


DESTRUCTIVE_PATTERNS = [
    re.compile(r"\brm\b", re.I),
    re.compile(r"\brmdir\b", re.I),
    re.compile(r"\bmv\b", re.I),
    re.compile(r"\bcp\b", re.I),
    re.compile(r"\bmkdir\b", re.I),
    re.compile(r"\btouch\b", re.I),
    re.compile(r"\bchmod\b", re.I),
    re.compile(r"\bchown\b", re.I),
    re.compile(r"\bchgrp\b", re.I),
    re.compile(r"\bln\b", re.I),
    re.compile(r"\btee\b", re.I),
    re.compile(r"\btruncate\b", re.I),
    re.compile(r"\bdd\b", re.I),
    re.compile(r"\bshred\b", re.I),
    re.compile(r"(^|[^<])>(?!>)"),
    re.compile(r">>"),
    re.compile(r"\bnpm\s+(install|uninstall|update|ci|link|publish)", re.I),
    re.compile(r"\byarn\s+(add|remove|install|publish)", re.I),
    re.compile(r"\bpnpm\s+(add|remove|install|publish)", re.I),
    re.compile(r"\bpip\s+(install|uninstall)", re.I),
    re.compile(r"\bapt(-get)?\s+(install|remove|purge|update|upgrade)", re.I),
    re.compile(r"\bbrew\s+(install|uninstall|upgrade)", re.I),
    re.compile(
        r"\bgit\s+"
        r"(add|commit|push|pull|merge|rebase|reset|checkout|branch\s+-[dD]|stash|"
        r"cherry-pick|revert|tag|init|clone)",
        re.I,
    ),
    re.compile(r"\bsudo\b", re.I),
    re.compile(r"\bsu\b", re.I),
    re.compile(r"\bkill\b", re.I),
    re.compile(r"\bpkill\b", re.I),
    re.compile(r"\bkillall\b", re.I),
    re.compile(r"\breboot\b", re.I),
    re.compile(r"\bshutdown\b", re.I),
    re.compile(r"\bsystemctl\s+(start|stop|restart|enable|disable)", re.I),
    re.compile(r"\bservice\s+\S+\s+(start|stop|restart)", re.I),
    re.compile(r"\b(vim?|nano|emacs|code|subl)\b", re.I),
]


SAFE_PATTERNS = [
    re.compile(r"^\s*cat\b"),
    re.compile(r"^\s*head\b"),
    re.compile(r"^\s*tail\b"),
    re.compile(r"^\s*less\b"),
    re.compile(r"^\s*more\b"),
    re.compile(r"^\s*grep\b"),
    re.compile(r"^\s*find\b"),
    re.compile(r"^\s*ls\b"),
    re.compile(r"^\s*pwd\b"),
    re.compile(r"^\s*echo\b"),
    re.compile(r"^\s*printf\b"),
    re.compile(r"^\s*wc\b"),
    re.compile(r"^\s*sort\b"),
    re.compile(r"^\s*uniq\b"),
    re.compile(r"^\s*diff\b"),
    re.compile(r"^\s*file\b"),
    re.compile(r"^\s*stat\b"),
    re.compile(r"^\s*du\b"),
    re.compile(r"^\s*df\b"),
    re.compile(r"^\s*tree\b"),
    re.compile(r"^\s*which\b"),
    re.compile(r"^\s*whereis\b"),
    re.compile(r"^\s*type\b"),
    re.compile(r"^\s*env\b"),
    re.compile(r"^\s*printenv\b"),
    re.compile(r"^\s*uname\b"),
    re.compile(r"^\s*whoami\b"),
    re.compile(r"^\s*id\b"),
    re.compile(r"^\s*date\b"),
    re.compile(r"^\s*cal\b"),
    re.compile(r"^\s*uptime\b"),
    re.compile(r"^\s*ps\b"),
    re.compile(r"^\s*top\b"),
    re.compile(r"^\s*htop\b"),
    re.compile(r"^\s*free\b"),
    re.compile(r"^\s*git\s+(status|log|diff|show|branch|remote|config\s+--get)", re.I),
    re.compile(r"^\s*git\s+ls-", re.I),
    re.compile(r"^\s*npm\s+(list|ls|view|info|search|outdated|audit)", re.I),
    re.compile(r"^\s*yarn\s+(list|info|why|audit)", re.I),
    re.compile(r"^\s*node\s+--version", re.I),
    re.compile(r"^\s*python\s+--version", re.I),
    re.compile(r"^\s*curl\s"),
    re.compile(r"^\s*wget\s+-O\s*-", re.I),
    re.compile(r"^\s*jq\b"),
    re.compile(r"^\s*sed\s+-n", re.I),
    re.compile(r"^\s*awk\b"),
    re.compile(r"^\s*rg\b"),
    re.compile(r"^\s*fd\b"),
    re.compile(r"^\s*bat\b"),
    re.compile(r"^\s*eza\b"),
]


def is_safe_command(command: str) -> bool:
    """命令在 plan 模式下是否允许执行（非破坏性且命中只读白名单）。"""
    destructive = any(pattern.search(command) for pattern in DESTRUCTIVE_PATTERNS)
    safe = any(pattern.search(command) for pattern in SAFE_PATTERNS)
    return not destructive and safe


def clean_step_text(text: str) -> str:
    cleaned = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(
        r"^(Use|Run|Execute|Create|Write|Read|Check|Verify|Update|Modify|Add|Remove|"
        r"Delete|Install)\s+(the\s+)?",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if len(cleaned) > 50:
        cleaned = cleaned[:47] + "..."
    return cleaned


def extract_todo_items(message: str) -> list[dict]:
    """从 `Plan:` 段落提取编号步骤。"""
    items: list[dict] = []
    header_match = re.search(r"\*{0,2}Plan:\*{0,2}\s*\n", message, re.I)
    if not header_match:
        return items
    plan_section = message[header_match.end() :]
    numbered_pattern = re.compile(r"^\s*(\d+)[.)]\s+\*{0,2}([^*\n]+)", re.M)
    for match in numbered_pattern.finditer(plan_section):
        text = match.group(2).strip()
        text = re.sub(r"\*{1,2}$", "", text).strip()
        if (
            len(text) > 5
            and not text.startswith("`")
            and not text.startswith("/")
            and not text.startswith("-")
        ):
            cleaned = clean_step_text(text)
            if len(cleaned) > 3:
                items.append({"step": len(items) + 1, "text": cleaned, "completed": False})
    return items


def extract_done_steps(message: str) -> list[int]:
    steps: list[int] = []
    for match in re.finditer(r"\[DONE:(\d+)\]", message, re.I):
        step = int(match.group(1))
        if step > 0:
            steps.append(step)
    return steps


def mark_completed_steps(text: str, items: list[dict]) -> int:
    """按 [DONE:n] 标记完成，返回标记数量。"""
    done_steps = extract_done_steps(text)
    for step in done_steps:
        for item in items:
            if item["step"] == step:
                item["completed"] = True
    return len(done_steps)
