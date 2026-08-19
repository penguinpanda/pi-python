from collections.abc import Callable

from ..models import Models
from ..providers import (
    deepseek_provider,
    faux_provider,
    google_provider,
    mistral_provider,
    azure_openai_responses_provider,
    github_copilot_provider,
    openrouter_provider,
    ant_ling_provider,
    openai_codex_provider,
    radius_provider,
    google_vertex_provider,
    amazon_bedrock_provider,
    baseten_provider,
    cerebras_provider,
    fireworks_provider,
    groq_provider,
    huggingface_provider,
    moonshotai_provider,
    moonshotai_cn_provider,
    nvidia_provider,
    together_provider,
    xiaomi_provider,
    zai_provider,
    zai_coding_cn_provider,
    xai_provider,
    opencode_provider,
    opencode_go_provider,
    xiaomi_token_plan_ams_provider,
    xiaomi_token_plan_cn_provider,
    xiaomi_token_plan_sgp_provider,
    cloudflare_ai_gateway_provider,
    cloudflare_workers_ai_provider,
    ollama_provider,
    openai_provider,
    qwen_provider,
    qwen_token_plan_cn_provider,
    qwen_token_plan_provider,
)
from ..types import Model


def _apply_codex_compat(model: Model) -> Model:
    """OpenAI Codex 模型统一开启 tool search（延迟导入避免循环）。"""
    from ..providers.openai_codex import apply_codex_compat

    return apply_codex_compat(model)


# 合并生成目录后按 provider 应用的模型后处理（生成目录元数据之上的运行时修正）。
_GENERATED_MODEL_POSTPROCESS: dict[str, Callable[[Model], Model]] = {
    "openai-codex": _apply_codex_compat,
}


def create_default_models() -> Models:
    """创建一个预加载内置 Provider 的 Models 实例。

    包含 OpenAI、DeepSeek、Qwen、Qwen Token Plan（国际站/中国站）、
    Ollama 与 Faux。
    """
    models = Models()
    models.add_provider(google_provider())
    models.add_provider(mistral_provider())
    models.add_provider(azure_openai_responses_provider())
    models.add_provider(github_copilot_provider())
    models.add_provider(openrouter_provider())
    models.add_provider(ant_ling_provider())
    models.add_provider(openai_codex_provider())
    models.add_provider(radius_provider())
    models.add_provider(google_vertex_provider())
    models.add_provider(amazon_bedrock_provider())
    models.add_provider(groq_provider())
    models.add_provider(together_provider())
    models.add_provider(cerebras_provider())
    models.add_provider(fireworks_provider())
    models.add_provider(nvidia_provider())
    models.add_provider(huggingface_provider())
    models.add_provider(baseten_provider())
    models.add_provider(moonshotai_provider())
    models.add_provider(xiaomi_provider())
    models.add_provider(zai_provider())
    models.add_provider(xai_provider())
    models.add_provider(moonshotai_cn_provider())
    models.add_provider(zai_coding_cn_provider())
    models.add_provider(opencode_provider())
    models.add_provider(opencode_go_provider())
    models.add_provider(xiaomi_token_plan_ams_provider())
    models.add_provider(xiaomi_token_plan_cn_provider())
    models.add_provider(xiaomi_token_plan_sgp_provider())
    models.add_provider(cloudflare_workers_ai_provider())
    models.add_provider(cloudflare_ai_gateway_provider())
    models.add_provider(openai_provider())
    models.add_provider(deepseek_provider())
    models.add_provider(qwen_provider())
    models.add_provider(qwen_token_plan_provider())
    models.add_provider(qwen_token_plan_cn_provider())
    models.add_provider(ollama_provider())
    # Faux 放在最后：不改变默认模型回退顺序（第一个可用模型仍是 openai 的）。
    models.add_provider(faux_provider().provider)
    # 接入生成目录：用 models/generated 的权威元数据覆盖已注册 provider
    # （同 id 覆盖，新 id 追加）。无 Python 实现的 provider（ant-ling、
    # openrouter 等）保持不可用，由对应 provider 模块提供后自动生效。
    _apply_generated_models(models)
    return models


def _apply_generated_models(models: Models) -> None:
    """把 models/generated 目录的模型元数据合并进已注册 provider。"""
    from ..models.generated import load_generated_models

    generated = load_generated_models()
    for provider_id, generated_models in generated.items():
        provider = models.get_provider(provider_id)
        if provider is None:
            continue
        merged = {model.id: model for model in provider.models}
        merged.update({model.id: model for model in generated_models})
        postprocess = _GENERATED_MODEL_POSTPROCESS.get(provider_id)
        if postprocess is not None:
            merged = {model_id: postprocess(model) for model_id, model in merged.items()}
        provider.models = list(merged.values())
