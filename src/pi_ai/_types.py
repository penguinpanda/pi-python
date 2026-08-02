"""pi_ai._types — 兼容 re-export 层（Deprecated）。

类型定义已迁移到 `pi_ai.types` 包（见 `pi_ai/types/`）。
本模块仅为保持 `from pi_ai._types import X` 的向后兼容而保留，
新代码请直接使用 `from pi_ai.types import X`。
"""

from typing import TYPE_CHECKING

from .types import *  # noqa: F401,F403
from .types import __all__  # noqa: F401

if TYPE_CHECKING:
    # 字符串前向引用需要（_types ↔ utils._event_stream 循环导入规避）。
    from .utils._event_stream import AssistantMessageEventStream
