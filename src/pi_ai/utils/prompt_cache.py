"""pi_ai.utils.prompt_cache — 提示缓存（Prompt Cache）参数解析。

对应 TypeScript 的移植：

    `packages/ai/src/api/openai-prompt-cache.ts`：
        OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH → OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH
        clampOpenAIPromptCacheKey          → clamp_openai_prompt_cache_key

    `packages/ai/src/api/openai-completions.ts` 的 resolveCacheRetention
    （配合 `utils/provider-env.ts` 的 getProviderEnvValue）：
        resolveCacheRetention              → resolve_cache_retention

用途：把 StreamOptions.session_id / cache_retention 解析为请求体参数：

    prompt_cache_key      （OpenAI 限制 64 字符）
    prompt_cache_retention （"long" 时发送 "24h"）
"""

from __future__ import annotations

import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types.common import CacheRetention, ProviderEnv

# OpenAI prompt_cache_key 的最大长度。
OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64


def clamp_openai_prompt_cache_key(key: str | None) -> str | None:
    """将 prompt_cache_key 截断到 OpenAI 的 64 字符限制。

    对齐 TS clampOpenAIPromptCacheKey：None 原样返回；
    超过 64 字符时按码位截断（对多字节字符安全）。
    """
    if key is None:
        return None
    chars = list(key)
    if len(chars) <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        return key
    return "".join(chars[:OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH])


def resolve_cache_retention(
    cache_retention: "CacheRetention | None" = None,
    env: "ProviderEnv | None" = None,
) -> "CacheRetention":
    """解析提示缓存保留策略（对齐 TS resolveCacheRetention）。

    优先级：
        ① 显式传入的 cache_retention
        ② 环境变量 PI_CACHE_RETENTION == "long" → "long"
        ③ 默认 "short"

    env 为 Provider 作用域环境变量覆盖（优先于 os.environ，对齐 TS
    getProviderEnvValue 的 env → process.env 顺序）。
    """
    if cache_retention:
        return cache_retention
    provider_env = env or {}
    # Provider 作用域覆盖优先：只要显式设置了 PI_CACHE_RETENTION，
    # 就按它的值解释（"long" → long，其余 → short），不再回退 os.environ。
    provider_value = provider_env.get("PI_CACHE_RETENTION")
    if provider_value is not None:
        return "long" if provider_value == "long" else "short"
    if os.environ.get("PI_CACHE_RETENTION") == "long":
        return "long"
    return "short"


__all__ = [
    "OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH",
    "clamp_openai_prompt_cache_key",
    "resolve_cache_retention",
]
