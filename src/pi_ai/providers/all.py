from ..models import Models
from ..providers import (
    deepseek_provider,
    faux_provider,
    ollama_provider,
    openai_provider,
    qwen_provider,
)

def create_default_models() -> Models:
    """ 创建一个预加载了OpenAI、DeepSeek、Ollama与Faux的Models实例。 """
    models = Models()
    models.add_provider(openai_provider())
    models.add_provider(deepseek_provider())
    models.add_provider(qwen_provider())
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
        provider.models = list(merged.values())
