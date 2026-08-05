"""Custom Provider Extension - register an OpenAI-compatible provider from env.

Python port of custom-provider-anthropic / custom-provider-gitlab-duo 的简化版：
用环境变量配置 base_url / api_key / model，通过 register_provider 接入。

环境变量：
- PI_CUSTOM_PROVIDER_BASE_URL
- PI_CUSTOM_PROVIDER_API_KEY（可省略，用 ${...} 引用）
- PI_CUSTOM_PROVIDER_MODEL
"""

import os

from pi_coding_agent import ExtensionAPI


def create_extension(pi: ExtensionAPI):
    base_url = os.environ.get("PI_CUSTOM_PROVIDER_BASE_URL")
    model_id = os.environ.get("PI_CUSTOM_PROVIDER_MODEL")
    if not base_url or not model_id:
        return
    api_key = os.environ.get("PI_CUSTOM_PROVIDER_API_KEY") or "${CUSTOM_PROVIDER_API_KEY}"
    pi.register_provider(
        "custom",
        {
            "name": "Custom OpenAI-compatible",
            "api": "openai-completions",
            "base_url": base_url,
            "api_key": api_key,
            "models": [{"id": model_id, "name": model_id, "reasoning": False}],
        },
    )
