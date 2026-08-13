# pi-ai — Unified LLM API

[English](README.en.md) | [中文](README.md)

Provider-abstraction LLM SDK, a Python port of [pi-mono/packages/ai](https://github.com/earendil-works/pi-mono).

Built-in providers: OpenAI, DeepSeek, Qwen, Qwen Token Plan, Ollama, Google, Mistral, Azure OpenAI, GitHub Copilot, OpenRouter, Ant Ling, OpenAI Codex, Google Vertex, AWS Bedrock, Radius (gateway dynamic catalog), and Faux (test). 13 more OpenAI-compatible providers (Groq, Together, Cerebras, Fireworks, xAI, NVIDIA, Hugging Face, Baseten, Moonshot, Xiaomi, Z.ai, ...) register via dynamic `/models` discovery.

## OAuth login

- **Browser flow** (PKCE + local callback server + manual-paste fallback): OpenAI Codex (port 1455), OpenRouter (ephemeral port), Radius (gateway discovery + port 1456)
- **Device-code flow**: xAI (SuperGrok/X Premium subscription), OpenAI Codex compatibility path
- Credentials persist to `~/.pi/agent/auth.json` (0600) with refresh expiry skew protection

## Quick Start

```python
import asyncio
from pi_ai import create_default_models, Context


async def main():
    models = create_default_models()
    model = models.get_model("deepseek", "deepseek-v4-flash")

    async for event in await models.stream(
        model,
        Context(messages=[{"role": "user", "content": "Hello!"}]),
    ):
        if event["type"] == "text_delta":
            print(event["delta"], end="", flush=True)


asyncio.run(main())
```

## Core concepts

- **Models** — provider registry + request dispatch + credential management
- **Provider** — base_url / auth / models / api kind configuration
- **API registry** — dispatch by `model.api`: openai-responses, openai-completions, pi-messages, google-generative-ai, google-vertex, mistral-conversations, azure-openai-responses, openai-codex-responses, bedrock-converse-stream
- **AssistantMessageEventStream** — 12 event types (start / text_* / thinking_* / toolcall_* / done / error)
- **OAuth** — browser PKCE loopback and device-code flows (Codex / GitHub Copilot / OpenRouter / xAI / Radius)

## Providers

| Provider | API | Notes |
|---|---|---|
| openai | Responses | `gpt-5-chat-latest`, `gpt-5.6-*` |
| deepseek | Responses / Completions | `deepseek-v4-flash` / `deepseek-v4-pro` |
| qwen | Completions | qwen-turbo/plus/max, qwen3-*, qwen-vl-* |
| ollama | Completions | local service, dynamic `/api/tags` discovery |
| google | Gemini REST SSE | strict tool sampling, advanced thinking |
| mistral | Completions | Mistral Conversations tool-call ID normalization |
| azure-openai-responses | Responses | Azure client + deployment mapping, `/openai/v1` base-path normalization |
| github-copilot | Completions | OAuth + API key dual auth |
| openrouter | Responses / Completions | 273 models from generated catalog |
| openai-codex | Codex Responses | WebSocket + SSE fallback, deferred tools |
| google-vertex | Vertex SSE | ADC credential fallback |
| amazon-bedrock | ConverseStream | thinking fields, cachePoint prompt caching, toolResult merging |
| radius | pi-messages | `GET /v1/config` dynamic catalog + subscription OAuth |
| faux | Completions | in-process test provider |

## Tests

```bash
uv run pytest src/pi_ai/tests/ -v
```

## License

MIT
