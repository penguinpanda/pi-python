"""一键比较：TS dump → Python dump → 逐字符比较。

运行（仓库根）:

    python src/pi_agent/tests/parity/compare_outputs.py

流程：先运行 TS 侧 dump（dump-system-prompt.ts，刷新 golden/），再运行
Python 侧 dump（dump_system_prompt.py，刷新 python_out/），最后逐字符比较
两侧输出。修改提示词（TS 或 Python）后运行本脚本即可立刻看到差异；
有差异时退出码为 1。

PI_PACKAGE_DIR 固定为 C:/pi-pkg（可用环境变量覆盖）；两侧 dump 均使用
同一个值，保证 "Pi documentation" 段的路径一致。
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"
OUT_DIR = HERE / "python_out"

DEFAULT_PACKAGE_DIR = "C:/pi-pkg"


def _run_dump(cmd: list[str], label: str) -> None:
    """运行一个 dump 脚本；失败只警告，不阻塞后续比较。"""
    env = dict(os.environ)
    env["PI_PACKAGE_DIR"] = os.environ.get("PI_PACKAGE_DIR") or DEFAULT_PACKAGE_DIR
    try:
        result = subprocess.run(
            cmd,
            cwd=HERE,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        print(f"[warn] {label} 不可运行（缺少 {exc.filename}），输出可能过期", file=sys.stderr)
        return
    for line in result.stdout.splitlines():
        print(line)
    if result.returncode != 0:
        print(f"[warn] {label} 失败 (exit {result.returncode})，输出可能过期", file=sys.stderr)
        print(result.stderr.strip(), file=sys.stderr)


def _compare() -> bool:
    """比较 golden/（TS）与 python_out/（PY），返回是否全部一致。"""
    if not GOLDEN_DIR.is_dir():
        print("[warn] golden/ 不存在，跳过比较（先运行 TS dump）", file=sys.stderr)
        return True
    all_ok = True
    for golden in sorted(GOLDEN_DIR.glob("*.txt")):
        out = OUT_DIR / golden.name
        if not out.exists():
            print(f"[missing] {golden.name}: python_out 缺输出")
            all_ok = False
            continue
        ts = golden.read_text(encoding="utf-8")
        py = out.read_text(encoding="utf-8")
        if ts == py:
            print(f"[ok] {golden.name}")
        else:
            all_ok = False
            print(f"[diff] {golden.name}")
            diff = difflib.unified_diff(
                ts.splitlines(),
                py.splitlines(),
                fromfile="TS",
                tofile="PY",
                lineterm="",
            )
            print("\n".join(diff))
    return all_ok


def main() -> None:
    _run_dump(
        ["node", "--experimental-strip-types", str(HERE / "dump-system-prompt.ts")], "TS dump"
    )
    _run_dump([sys.executable, str(HERE / "dump_system_prompt.py")], "Python dump")
    if not _compare():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
