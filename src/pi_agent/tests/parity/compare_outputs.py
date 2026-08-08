"""一键比较：Python dump → 与 golden 逐字符比较。

运行（仓库根）:

    python src/pi_agent/tests/parity/compare_outputs.py

流程：先运行 Python 侧 dump（dump_system_prompt.py，刷新 python_out/），
再与已入库的 golden/ 逐字符比较。golden 需在 pi TS mono-repo 中生成后拷入
（见 README.md）。修改 Python 提示词后运行本脚本即可立刻看到与 TS golden
的差异；有差异时退出码为 1。

PI_PACKAGE_DIR 固定为 C:/pi-pkg（可用环境变量覆盖）；dump 与测试均使用
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
    """比较 python_out/（PY）与 golden/（TS），返回是否全部一致。"""
    all_ok = True
    for out in sorted(OUT_DIR.glob("*.txt")):
        golden = GOLDEN_DIR / out.name
        if not golden.exists():
            print(f"[missing] {out.name}: golden 缺失", file=sys.stderr)
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
    _run_dump([sys.executable, str(HERE / "dump_system_prompt.py")], "Python dump")
    if not _compare():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
