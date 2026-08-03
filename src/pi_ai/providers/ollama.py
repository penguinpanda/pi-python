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

    ① 定义本地已安装的模型（静态目录 + 运行时动态发现）

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

import os

import httpx

from ..types import Model, ModelCost
from ..provider import create_provider, Provider, RefreshModelsContext


# Ollama 服务根地址（原生 API /api/tags 使用，不带 /v1）。
#
# 使用 127.0.0.1 而非 localhost：
# httpx 默认读取 Windows 系统代理（trust_env=True），
# localhost 可能被本地代理拦截导致 503。
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def _ollama_base_url() -> str:
    """Ollama 服务根地址，可用环境变量 OLLAMA_BASE_URL 覆盖。

    容器内运行时可指向宿主机（如 http://host.docker.internal:11434）。
    """
    return os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)


# ------------------------------------------------------
# 静态模型目录（丰富元数据来源）。
#
# 来源：`ollama list`。
#
# Ollama 的模型是动态 pull 的，
# 安装/卸载后与 `ollama list` 可能不一致。
# 运行时动态发现见 discover_ollama_models()：
# 已知模型复用本目录的元数据，未知模型合成默认元数据。
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
        max_tokens=8192,
        context_window=131072,
        reasoning=True,
        cost=ModelCost(),  # 本地运行，无费用
    ),

    # Qwen3 30B-A3B（MoE 版本）
    Model(
        id="qwen3:30b-a3b",
        provider="ollama",
        api="openai-completions",
        name="Qwen3 30B A3B",
        input=["text"],
        output=["text"],
        max_tokens=8192,
        context_window=131072,
        reasoning=True,
        cost=ModelCost(),
    ),

    # Qwen3 14B Abliterated（社区去审查版）
    Model(
        id="richardyoung/qwen3-14b-abliterated:Q5_K_M",
        provider="ollama",
        api="openai-completions",
        name="Qwen3 14B Abliterated",
        input=["text"],
        output=["text"],
        max_tokens=8192,
        context_window=131072,
        reasoning=True,
        cost=ModelCost(),
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
        max_tokens=32768,
        context_window=131072,
        reasoning=True,
        cost=ModelCost(),
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
        max_tokens=4096,
        context_window=131072,
        cost=ModelCost(),
    ),

    # Qwen 2.5 7B Instruct
    Model(
        id="qwen2.5:7b-instruct-q8_0",
        provider="ollama",
        api="openai-completions",
        name="Qwen 2.5 7B Instruct",
        input=["text"],
        output=["text"],
        max_tokens=8192,
        context_window=131072,
        cost=ModelCost(),
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
        max_tokens=8192,
        context_window=65536,
        reasoning=True,
        cost=ModelCost(),
    ),
]


# ------------------------------------------------------
# 运行时动态发现
# ------------------------------------------------------


def _default_model(name: str) -> Model:
    """为静态目录中不存在的模型合成默认元数据。"""
    return Model(
        id=name,
        provider="ollama",
        api="openai-completions",
        name=name,
        input=["text"],
        output=["text"],
        max_tokens=8192,
        cost=ModelCost(),  # 本地运行，无费用
    )


def _merge_ollama_models(names: list[str]) -> list[Model]:
    """将 /api/tags 返回的模型名与静态元数据合并。

    已知模型保留静态目录的丰富元数据（thinking / 工具 / 图片）；
    未知模型（新 pull 的）合成默认元数据。
    返回顺序与 /api/tags 一致。
    """
    static_by_id = {m.id: m for m in OLLAMA_MODELS}
    result: list[Model] = []
    for name in names:
        static = static_by_id.get(name)
        result.append(static if static is not None else _default_model(name))
    return result


async def discover_ollama_models(
    base_url: str | None = None,
    timeout: float = 1.0,
) -> list[Model] | None:
    """运行时发现 Ollama 已安装的模型（GET /api/tags）。

    返回按 /api/tags 顺序、合并静态元数据的模型列表，
    使 `ollama pull` / `ollama rm` 后模型列表实时同步。

    失败（未运行 / 超时 / 非 200）返回 None，
    调用方可回退到静态 OLLAMA_MODELS。
    """
    base_url = base_url or _ollama_base_url()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url}/api/tags")
        if resp.status_code != 200:
            return None
        data = resp.json()
        names = [
            item.get("name", "")
            for item in data.get("models", [])
            if item.get("name")
        ]
        return _merge_ollama_models(names)
    except Exception:
        # 任何网络/解析失败都不阻断调用方，回退静态列表。
        return None


async def _fetch_ollama_models(context: RefreshModelsContext) -> list[Model]:
    """refreshModels 用的抓取实现：失败抛异常（由 Models.refresh 收集）。"""
    discovered = await discover_ollama_models()
    if discovered is None:
        raise RuntimeError("Ollama model discovery failed (GET /api/tags)")
    return discovered


def ollama_provider(models: list[Model] | None = None) -> Provider:
    """
    创建并返回一个 Ollama Provider。

    Provider 已预先配置：

        • Provider ID

        • 模型列表

        • API 类型（OpenAI Compatible Completions）

        • Base URL

    Ollama 本地服务默认不需要 API Key，
    因此不配置认证（auth=None）。

    models：

        默认使用静态 OLLAMA_MODELS。
        传入 discover_ollama_models() 的结果可启用运行时动态发现。

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

        models=models if models is not None else OLLAMA_MODELS,

        # Ollama 提供 OpenAI Chat Completions 兼容接口。
        api_kind="completions",

        # Ollama OpenAI Compatible API 地址。
        #
        # 使用 127.0.0.1 而非 localhost，
        # 避免 httpx 走 Windows 系统代理。
        base_url=f"{_ollama_base_url()}/v1",
        fetch_models=_fetch_ollama_models,
    )
