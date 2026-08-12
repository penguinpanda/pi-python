"""首次启动向导：配置初始 provider 与 API key。"""

from __future__ import annotations

import asyncio
import os


def are_experimental_features_enabled() -> bool:
    """PI_EXPERIMENTAL 门控（对齐 TS areExperimentalFeaturesEnabled）。"""
    return os.environ.get("PI_EXPERIMENTAL") == "1"


def should_run_first_time_setup() -> bool:
    """对齐 TS shouldRunFirstTimeSetup：experimental + 默认 agent 目录 +
    settings.json 不存在。"""
    if not are_experimental_features_enabled():
        return False
    if os.environ.get("PI_CODING_AGENT_DIR"):
        return False
    from ._config import get_settings_path

    try:
        return not get_settings_path().exists()
    except Exception:
        return False


async def run_first_time_setup(auth_store) -> int:
    """交互式配置：选择 provider → 输入 API key → 保存到 auth.json。"""
    providers = [("openai", "OpenAI"), ("deepseek", "DeepSeek")]
    print("Welcome to pi!")
    print("Let's configure an API provider.\n")
    print("Providers:")
    for index, (provider_id, name) in enumerate(providers, 1):
        print(f"  {index}. {name} ({provider_id})")
    while True:
        raw = input(f"Enter number (1-{len(providers)}): ").strip()
        try:
            provider_id, display_name = providers[int(raw) - 1]
            break
        except (ValueError, IndexError):
            print("Invalid selection.")

    api_key = input(f"Enter your {display_name} API key: ").strip()
    if not api_key:
        print("No API key provided. Setup skipped.")
        return 1

    from pi_ai.auth import ApiKeyCredential

    async def _set(_current):
        return ApiKeyCredential(type="api_key", key=api_key)

    await auth_store.modify(provider_id, _set)
    print(f"\nCredentials saved to {auth_store.path}")

    # experimental 门控的 analytics 同意询问（对齐 TS showAnalyticsConsent）。
    if are_experimental_features_enabled():
        _ask_analytics_consent()

    print("You can now run: pi-python -p 'hello'")
    return 0


def _ask_analytics_consent() -> None:
    """询问 enableAnalytics 并写入全局设置（对齐 TS showAnalyticsConsent）。"""
    answer = input("\nShare anonymous usage analytics to improve pi? (y/N): ").strip().lower()
    if answer not in ("y", "yes"):
        return
    from .settings_manager import SettingsManager

    try:
        manager = SettingsManager.create(os.getcwd(), project_trusted=False)
        manager.set_global_setting("enableAnalytics", True)
    except Exception:
        print("Could not save analytics preference.")


def run_first_time_setup_sync(auth_store) -> int:
    """同步包装（CLI 用）。"""
    return asyncio.run(run_first_time_setup(auth_store))


__all__ = ["run_first_time_setup", "run_first_time_setup_sync"]
