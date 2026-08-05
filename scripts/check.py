"""本地运行 GitHub Actions 同款检查（ruff / mypy / pytest）。

用法（仓库根目录）:
    python scripts/check.py
    uv run --no-sync scripts/check.py

任一检查失败立即退出，返回对应非零码。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIRS = [
    "src/pi_ai",
    "src/pi_agent",
    "src/pi_coding_agent",
    "src/pi_tui",
    "src/pi_protocol",
    "src/pi_storage",
    "src/pi_server",
    "src/pi_evals",
]

STEPS: list[tuple[str, list[str]]] = [
    ("Ruff lint", ["ruff", "check", "."]),
    ("Ruff format check", ["ruff", "format", "--check", "."]),
    ("Type check", ["mypy", *SRC_DIRS]),
    (
        "Tests with coverage",
        [
            "pytest",
            "-q",
            "-k",
            "not live and not oauth and not device",
            "--cov=pi_ai",
            "--cov=pi_agent",
            "--cov=pi_coding_agent",
            "--cov-report=term-missing",
        ],
    ),
]


def _run_step(name: str, args: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    env = dict(os.environ)
    # 默认 uv 缓存损坏/不可写时（Windows 偶发）回退到临时目录。
    env.setdefault("UV_CACHE_DIR", os.path.join(tempfile.gettempdir(), "pi-uv-cache"))
    try:
        result = subprocess.run(
            ["uv", "run", "--no-sync", *args],
            cwd=REPO_ROOT,
            env=env,
        )
    except FileNotFoundError:
        print(
            "uv not found on PATH. Install uv (https://docs.astral.sh/uv/) or run the checks manually.",
            file=sys.stderr,
        )
        sys.exit(1)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    for name, args in STEPS:
        _run_step(name, args)
    print("\nAll checks passed!", flush=True)


if __name__ == "__main__":
    main()
