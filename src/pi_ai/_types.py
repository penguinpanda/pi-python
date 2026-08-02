"""pi_ai._types — 兼容 re-export 层（Deprecated）。

类型定义已迁移到 `pi_ai.types` 包（见 `pi_ai/types/`）。
本模块仅为保持 `from pi_ai._types import X` 的向后兼容而保留，
新代码请直接使用 `from pi_ai.types import X`。
"""

import sys

from typing import TYPE_CHECKING


def _warn_external_deprecation() -> None:
    """仅对包外直接导入发出 DeprecationWarning（包内迁移期不打扰）。"""
    try:
        frame = sys._getframe(2)
        for _ in range(8):
            name = frame.f_globals.get("__name__") or ""
            if name == "pi_ai" or name.startswith("pi_ai."):
                return
            frame = frame.f_back
            if frame is None:
                break
    except Exception:
        return
    import warnings

    warnings.warn(
        "pi_ai._types is deprecated; use pi_ai.types instead.",
        DeprecationWarning,
        stacklevel=3,
    )


_warn_external_deprecation()

from .types import *  # noqa: F401,F403
from .types import __all__  # noqa: F401

if TYPE_CHECKING:
    # 字符串前向引用需要（_types ↔ utils._event_stream 循环导入规避）。
    from .utils._event_stream import AssistantMessageEventStream
