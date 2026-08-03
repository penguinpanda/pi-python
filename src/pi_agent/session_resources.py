"""pi_agent.session_resources — 兼容 re-export（Deprecated）。

实现已迁移到 `pi_ai.session_resources`（对齐 TS packages/ai/src/session-resources.ts）。
新代码请直接使用 `from pi_ai.session_resources import ...`。
"""

import warnings

warnings.warn(
    "pi_agent.session_resources is deprecated; use pi_ai.session_resources instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pi_ai.session_resources import *  # noqa: F401,F403
from pi_ai.session_resources import __all__  # noqa: F401
