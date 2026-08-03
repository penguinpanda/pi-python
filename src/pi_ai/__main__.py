"""python -m pi_ai 入口（对齐 TS packages/ai/src/cli.ts）。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
