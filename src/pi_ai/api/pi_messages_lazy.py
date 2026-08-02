"""pi-messages 懒加载入口（对齐 TS pi-messages.lazy.ts）。"""

import importlib

from .lazy import lazy_api


def pi_messages_api():
    """返回 ProviderStreams；模块在首次 stream 调用时加载。"""

    async def _load():
        return importlib.import_module(".pi_messages", __package__)

    return lazy_api(_load)


__all__ = ["pi_messages_api"]
