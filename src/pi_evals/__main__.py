"""python -m pi_evals 入口（等价于 console script pi-evals）。"""

from __future__ import annotations

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
