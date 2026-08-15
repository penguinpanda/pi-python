"""pi_ai.api.simple_options — 请求参数构建辅助（移植 TS packages/ai/src/api/simple-options.ts）。

当前只移植 max_tokens 上下文收敛部分：

    clamp_max_tokens_to_context()

buildBaseOptions 的其余字段由各 API 文件（completions.py / responses.py）
按需内联构建，暂未独立成函数。
"""

from __future__ import annotations

from ..types import Context, Model
from ..utils.estimate import estimate_context_tokens

# 为模型回复预留的上下文 token 数。
CONTEXT_SAFETY_TOKENS = 4096

# max_tokens 的最小值。
MIN_MAX_TOKENS = 1


def clamp_max_tokens_to_context(model: Model, context: Context, max_tokens: int) -> int:
    """把 max_tokens 收敛到模型上下文窗口内（预留 CONTEXT_SAFETY_TOKENS）。

    - 模型未声明 context_window（<= 0）：返回 max(MIN_MAX_TOKENS, max_tokens)
    - 否则：available = context_window - 估算上下文 - 安全余量；
      max_tokens 不超过 available（且不小于 MIN_MAX_TOKENS）
    """
    if model.context_window <= 0:
        return max(MIN_MAX_TOKENS, max_tokens)

    available = (
        model.context_window - estimate_context_tokens(context).tokens - CONTEXT_SAFETY_TOKENS
    )
    # 两条分支都收敛到 [MIN_MAX_TOKENS, +∞)：负/零 max_tokens 不得透传给
    # provider（会被拒绝），长上下文时也不得静默把输出压到 0。
    return min(max(MIN_MAX_TOKENS, max_tokens), max(MIN_MAX_TOKENS, available))


__all__ = [
    "CONTEXT_SAFETY_TOKENS",
    "MIN_MAX_TOKENS",
    "clamp_max_tokens_to_context",
]
