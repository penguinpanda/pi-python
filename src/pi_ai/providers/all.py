from ..models import Models
from ..providers import (
    deepseek_provider,
    ollama_provider,
    openai_provider,
)

def create_default_models() -> Models:
    """ 创建一个预加载了OpenAI和DeepSeek的Models实例。 """
    models = Models()
    models.add_provider(openai_provider())
    models.add_provider(deepseek_provider())
    models.add_provider(ollama_provider())
    return models