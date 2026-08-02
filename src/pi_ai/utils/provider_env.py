"""Provider 环境值查询（对齐 TS utils/provider-env.ts）。"""

import os

from ..types.common import ProviderEnv


def get_provider_env_value(name: str, env: ProviderEnv | None = None) -> str | None:
    """按"显式 env 覆盖 → os.environ"顺序取值；空值视为未设置。

    TS 侧另有 Bun 沙箱的 /proc/self/environ 回退，Python 不适用。
    """
    if env:
        value = env.get(name)
        if value:
            return value
    value = os.environ.get(name)
    return value if value else None


__all__ = ["get_provider_env_value"]
