"""Merge and Resolve - fetch/merge upstream after each turn, report conflicts.

Python port of git-merge-and-resolve.ts（简化）。
"""

from pathlib import Path

from pi_coding_agent import ExtensionAPI


async def _find_conflicts(pi: ExtensionAPI, cwd: str) -> list[dict]:
    try:
        result = await pi.exec("git", ["diff", "--name-only", "--diff-filter=U"], {"cwd": cwd})
    except Exception:
        return []
    if result.get("exit_code") != 0 or not str(result.get("output", "")).strip():
        return []
    blocks: list[dict] = []
    for file in str(result.get("output", "")).splitlines():
        try:
            lines = (Path(cwd) / file).read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        start = sep = None
        for index, line in enumerate(lines, 1):
            if line.startswith("<<<<<<<"):
                start = index
                sep = None
            elif line.startswith("=======") and start is not None:
                sep = index
            elif line.startswith(">>>>>>>") and start is not None and sep is not None:
                blocks.append(
                    {"file": file, "startLine": start, "separatorLine": sep, "endLine": index}
                )
                start = sep = None
    return blocks


def create_extension(pi: ExtensionAPI):
    async def on_agent_end(event, ctx):
        try:
            git_dir = await pi.exec("git", ["rev-parse", "--git-dir"])
        except Exception:
            return
        if git_dir.get("exit_code") != 0:
            return
        try:
            merge_head = await pi.exec("git", ["rev-parse", "MERGE_HEAD"])
        except Exception:
            return
        if merge_head.get("exit_code") != 0:
            # 工作区干净才尝试合并 upstream。
            try:
                status = await pi.exec("git", ["status", "--porcelain"])
                upstream = await pi.exec(
                    "git", ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
                )
            except Exception:
                return
            if str(status.get("output", "")).strip() or upstream.get("exit_code") != 0:
                return
            ref = str(upstream.get("output", "")).strip()
            remote = ref.split("/", 1)[0]
            ctx.ui.notify(f"git-merge-and-resolve: fetching {remote}, merging {ref}", "info")
            try:
                await pi.exec("git", ["fetch", remote])
                merge = await pi.exec("git", ["merge", ref])
            except Exception:
                return
            if merge.get("exit_code") != 0:
                pass  # 冲突由下面统一上报
        conflicts = await _find_conflicts(pi, ctx.cwd)
        if conflicts:
            lines = ["Merge with conflicts:"]
            for block in conflicts:
                lines.append(
                    f"  {block['file']}:{block['startLine']}-{block['endLine']} "
                    f"(ours {block['startLine'] + 1}-{block['separatorLine'] - 1}, "
                    f"theirs {block['separatorLine'] + 1}-{block['endLine'] - 1})"
                )
            lines.append("Resolve these conflicts.")
            pi.send_user_message("\n".join(lines), {"deliverAs": "followUp"})

    pi.on("agent_end", on_agent_end)
