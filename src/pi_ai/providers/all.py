from ..models import Models
from ..providers import (
    deepseek_provider,
    faux_provider,
    ollama_provider,
    openai_provider,
)

def create_default_models() -> Models:
    """ 创建一个预加载了OpenAI、DeepSeek、Ollama与Faux的Models实例。 """
    models = Models()
    models.add_provider(openai_provider())
    models.add_provider(deepseek_provider())
    models.add_provider(ollama_provider())
    # Faux 放在最后：不改变默认模型回退顺序（第一个可用模型仍是 openai 的）。
    models.add_provider(faux_provider().provider)
    return models