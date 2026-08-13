"""Coding-agent 全局 HTTP dispatcher 配置。

对齐 TS core/http-dispatcher.ts 的 Python 最小等价实现。Python 的 provider
大多直接实例化 `httpx.AsyncClient`，因此这里提供：

- 全局代理环境变量注入（httpx 默认 trust_env 会读取）；
- 默认 idle timeout 常量与解析；
- `make_http_client()` 工厂，让本项目可控制的 HTTP 客户端统一采用配置。
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_HTTP_IDLE_TIMEOUT_MS = 300_000

HTTP_IDLE_TIMEOUT_CHOICES = (
    (30_000, "30 sec"),
    (60_000, "1 min"),
    (120_000, "2 min"),
    (300_000, "5 min"),
    (0, "disabled"),
)


def parse_http_idle_timeout_ms(value: Any) -> int | None:
    """解析 idle timeout，兼容字符串、数字和 disabled。"""
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() == "disabled":
            return 0
        if not trimmed:
            return None
        try:
            value = float(trimmed)
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return int(value)


def format_http_idle_timeout_ms(timeout_ms: int) -> str:
    for choice_ms, label in HTTP_IDLE_TIMEOUT_CHOICES:
        if timeout_ms == choice_ms:
            return label
    return f"{timeout_ms / 1000:g} sec"


def apply_http_proxy_settings(http_proxy: str | None) -> None:
    """写入全局代理环境变量，沿用 httpx 的 trust_env 行为。"""
    proxy = (http_proxy or "").strip()
    if not proxy:
        return
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)


_configured_timeout_ms = DEFAULT_HTTP_IDLE_TIMEOUT_MS
_original_async_client_init = httpx.AsyncClient.__init__
_async_client_init_patched = False
_original_default_timeout = (
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect,
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read,
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write,
    httpx._config.DEFAULT_TIMEOUT_CONFIG.pool,
)


def _as_timeout(value: Any) -> httpx.Timeout:
    if isinstance(value, httpx.Timeout):
        return value
    return httpx.Timeout(value)


def _limit_timeout(timeout: httpx.Timeout, timeout_seconds: float | None) -> httpx.Timeout:
    """把显式 timeout 的未禁用字段限制到全局 idle timeout。"""
    if timeout_seconds is None:
        return httpx.Timeout(timeout=None)
    values: list[float | None] = []
    for field in (timeout.connect, timeout.read, timeout.write, timeout.pool):
        values.append(None if field is None else min(float(field), timeout_seconds))
    return httpx.Timeout(
        connect=values[0],
        read=values[1],
        write=values[2],
        pool=values[3],
    )


def _apply_global_default_timeout(timeout_ms: int) -> None:
    """给未显式传 timeout 的 httpx.AsyncClient 设置全局默认空闲超时。"""
    timeout_seconds = None if timeout_ms == 0 else timeout_ms / 1000.0
    default = httpx._config.DEFAULT_TIMEOUT_CONFIG
    default.connect = timeout_seconds
    default.read = timeout_seconds
    default.write = timeout_seconds
    default.pool = timeout_seconds


def _install_async_client_timeout_patch() -> None:
    """让所有 `httpx.AsyncClient(...)` 实例都遵守全局 idle timeout。"""
    global _original_async_client_init, _async_client_init_patched
    if _async_client_init_patched:
        return

    def _patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        sentinel = object()
        explicit = kwargs.pop("timeout", sentinel)
        timeout_seconds = None if _configured_timeout_ms == 0 else _configured_timeout_ms / 1000.0
        if explicit is sentinel or explicit is None:
            kwargs["timeout"] = (
                httpx.Timeout(timeout=None)
                if timeout_seconds is None
                else httpx.Timeout(timeout_seconds)
            )
        else:
            kwargs["timeout"] = _limit_timeout(_as_timeout(explicit), timeout_seconds)
        _original_async_client_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = _patched_init  # type: ignore[method-assign]
    _async_client_init_patched = True


def reset_http_dispatcher() -> None:
    """恢复 httpx 默认状态（测试隔离用）。"""
    global _configured_timeout_ms, _async_client_init_patched
    if _async_client_init_patched:
        httpx.AsyncClient.__init__ = _original_async_client_init  # type: ignore[method-assign]
        _async_client_init_patched = False
    _configured_timeout_ms = DEFAULT_HTTP_IDLE_TIMEOUT_MS
    default = httpx._config.DEFAULT_TIMEOUT_CONFIG
    default.connect = _original_default_timeout[0]
    default.read = _original_default_timeout[1]
    default.write = _original_default_timeout[2]
    default.pool = _original_default_timeout[3]


def configure_http_dispatcher(timeout_ms: int = DEFAULT_HTTP_IDLE_TIMEOUT_MS) -> None:
    """配置全局 HTTP 默认值并启用系统代理。"""
    global _configured_timeout_ms
    normalized = parse_http_idle_timeout_ms(timeout_ms)
    if normalized is None:
        raise ValueError(f"Invalid HTTP idle timeout: {timeout_ms}")
    _configured_timeout_ms = normalized
    _apply_global_default_timeout(normalized)
    _install_async_client_timeout_patch()
    # 不覆盖用户显式设置的代理；httpx 默认 trust_env 会读取这两个变量。
    os.environ.setdefault("HTTP_PROXY", os.environ.get("HTTP_PROXY", ""))
    os.environ.setdefault("HTTPS_PROXY", os.environ.get("HTTPS_PROXY", ""))


def get_configured_http_timeout_ms() -> int:
    return _configured_timeout_ms


def make_http_client(*, timeout_ms: int | None = None, **kwargs: Any) -> httpx.AsyncClient:
    """创建使用全局 idle timeout 配置的 httpx.AsyncClient。"""
    timeout = httpx.Timeout(
        (get_configured_http_timeout_ms() if timeout_ms is None else timeout_ms) / 1000.0
    )
    kwargs.setdefault("timeout", timeout)
    kwargs.setdefault("trust_env", True)
    return httpx.AsyncClient(**kwargs)


__all__ = [
    "DEFAULT_HTTP_IDLE_TIMEOUT_MS",
    "HTTP_IDLE_TIMEOUT_CHOICES",
    "parse_http_idle_timeout_ms",
    "format_http_idle_timeout_ms",
    "apply_http_proxy_settings",
    "configure_http_dispatcher",
    "reset_http_dispatcher",
    "get_configured_http_timeout_ms",
    "make_http_client",
]
