"""
Ollama Provider。

=========================================================
模块职责
=========================================================

本模块负责注册 Ollama Provider。

Ollama 是本地模型运行时，
提供 OpenAI 兼容接口：

    http://localhost:11434/v1/chat/completions

因此复用现有 completions API 实现，
不需要新写 HTTP 层。

主要包括：

    ① 定义本地已安装的模型

    ② 配置 API 类型（completions）

    ③ 配置 Base URL

本地服务默认不需要 API Key，
因此 auth 使用 None（见 provider.py）。

最终通过：

    ollama_provider()

创建一个可直接使用的 Provider。

整体关系：

        OLLAMA_MODELS
                │
                ▼
        create_provider()
                │
                ▼
           Provider
"""

from .._types import Model
from ..provider import create_provider, Provider


# ------------------------------------------------------
# 本地已安装的模型列表。
#
# 来源：`ollama list`。
#
# Ollama 的模型是动态 pull 的，
# 如果之后安装/卸载了模型，
# 需要同步更新本列表
# （或者改为运行时调用 /api/tags 动态生成）。
#
# 模型 ID 必须与 `ollama list` 的 NAME 完全一致，
# 因为它会原样作为 model 参数发送给 API。
# ------------------------------------------------------
OLLAMA_MODELS: list[Model] = [

    # Qwen3 30B
    #
    # 通用对话模型。
    #
    # 特点：
    #
    # • 支持 Thinking
    # • 支持 Tool Calling
    # • 文本输入
    Model(
        id="qwen3:30b",
        provider="ollama",
        api="openai-completions",
        name="Qwen3 30B",
        input=["text"],
        output=["text"],
        maxTokens=8192,
        thinking=True,
        supportsToolCalling=True,
        supportsImages=False,
        cost={},  # 本地运行，无费用
    ),

    # Qwen3 30B-A3B（MoE 版本）
    Model(
        id="qwen3:30b-a3b",
        provider="ollama",
        api="openai-completions",
        name="Qwen3 30B A3B",
        input=["text"],
        output=["text"],
        maxTokens=8192,
        thinking=True,
        supportsToolCalling=True,
        supportsImages=False,
        cost={},
    ),

    # Qwen3 14B Abliterated（社区去审查版）
    Model(
        id="richardyoung/qwen3-14b-abliterated:Q5_K_M",
        provider="ollama",
        api="openai-completions",
        name="Qwen3 14B Abliterated",
        input=["text"],
        output=["text"],
        maxTokens=8192,
        thinking=True,
        supportsToolCalling=True,
        supportsImages=False,
        cost={},
    ),

    # GPT-OSS 20B
    #
    # OpenAI 开源权重模型。
    #
    # 特点：
    #
    # • 支持 Thinking
    # • 支持 Tool Calling
    Model(
        id="gpt-oss:20b",
        provider="ollama",
        api="openai-completions",
        name="GPT-OSS 20B",
        input=["text"],
        output=["text"],
        maxTokens=32768,
        thinking=True,
        supportsToolCalling=True,
        supportsImages=False,
        cost={},
    ),

    # Llama 3.2 Vision
    #
    # 多模态模型。
    #
    # 特点：
    #
    # • 支持图片输入
    # • 支持 Tool Calling
    Model(
        id="llama3.2-vision:latest",
        provider="ollama",
        api="openai-completions",
        name="Llama 3.2 Vision",
        input=["text", "image"],
        output=["text"],
        maxTokens=4096,
        thinking=False,
        supportsToolCalling=True,
        supportsImages=True,
        cost={},
    ),

    # Qwen 2.5 7B Instruct
    Model(
        id="qwen2.5:7b-instruct-q8_0",
        provider="ollama",
        api="openai-completions",
        name="Qwen 2.5 7B Instruct",
        input=["text"],
        output=["text"],
        maxTokens=8192,
        thinking=False,
        supportsToolCalling=True,
        supportsImages=False,
        cost={},
    ),

    # DeepSeek R1 14B
    #
    # 推理模型。
    #
    # 特点：
    #
    # • 支持 Thinking
    # • 不支持 Tool Calling
    Model(
        id="deepseek-r1:14b",
        provider="ollama",
        api="openai-completions",
        name="DeepSeek R1 14B",
        input=["text"],
        output=["text"],
        maxTokens=8192,
        thinking=True,
        supportsToolCalling=False,
        supportsImages=False,
        cost={},
    ),
]


def ollama_provider() -> Provider:
    """
    创建并返回一个 Ollama Provider。

    Provider 已预先配置：

        • Provider ID

        • 模型列表

        • API 类型（OpenAI Compatible Completions）

        • Base URL

    Ollama 本地服务默认不需要 API Key，
    因此不配置认证（auth=None）。

    调用者通常只需要：

        provider = ollama_provider()

    即可完成 Provider 注册。

    默认地址使用 127.0.0.1 而不是 localhost：

    httpx 默认读取 Windows 系统代理（trust_env=True），
    localhost 可能被本地代理拦截导致 503；
    127.0.0.1 是同一地址的直连形式，可绕过该问题。

    如果 Ollama 不在 127.0.0.1:11434，
    可以修改 base_url（如 OLLAMA_HOST 指定的地址）。
    """
    return create_provider(
        id="ollama",
        name="Ollama",

        # 本地服务无需 API Key。
        auth=None,

        models=OLLAMA_MODELS,

        # Ollama 提供 OpenAI Chat Completions 兼容接口。
        api_kind="completions",

        # Ollama OpenAI Compatible API 地址。
        #
        # 使用 127.0.0.1 而非 localhost，
        # 避免 httpx 走 Windows 系统代理。
        base_url="http://127.0.0.1:11434/v1",
        # base_url="http://localhost/v1",
    )
