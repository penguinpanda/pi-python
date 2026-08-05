# 自定义 Provider（Python 移植）

pi-python 支持两层自定义 provider：底层 `pi_ai` 注册 API 实现，应用层 `ModelRuntime` 注册 provider 配置（含认证、模型、OAuth、自定义流）。

## 快速参考

应用层注册（扩展内）：

```python
pi.register_provider(
    "my-provider",
    {
        "name": "My Provider",
        "api": "openai-completions",  # 复用内置流实现
        "base_url": "https://api.example.com/v1",
        "api_key": "${MY_PROVIDER_API_KEY}",  # 环境变量引用
        "headers": {"X-Custom": "1"},
        "models": [{"id": "my-model", "name": "My Model", "reasoning": True}],
    },
)
```

`pi.unregister_provider("my-provider")` 注销。

## 底层：`pi_ai` API 注册表

`src/pi_ai/api/api_provider_registry.py`：

- `ApiProvider(api, stream, streamSimple, source_id)`：`stream` 是 `(model, context, options) -> AssistantMessageEventStream` 的流函数
- `register_api_provider(provider, source_id)` / `get_api_provider(api)` / `get_api_providers()` / `unregister_api_providers(source_id)` / `reset_api_providers()`
- 内置条目：`openai-completions`、`openai-responses`、`pi-messages`
- `stream(model, context, options)` 顶层分发：按 `model.api` 找条目；未显式传 `api_key` 时按 provider 注入环境变量（`_with_env_api_key`）

`src/pi_ai/provider.py` 的 `Provider` 持有配置（id、base_url、auth、models、refresh_models），`stream()` 里解析凭证（存储凭证 > 配置 key > 环境变量）后把 `api_key/base_url` 注入 options 再调用注册表。

## 应用层：`ModelRuntime`

`src/pi_coding_agent/model_runtime.py`：

- `register_provider(provider_id, config: ProviderConfigInput)`：config 键为 `name` / `base_url` / `api_key` / `api` / `headers` / `auth_header` / `models` / `oauth` / `stream_simple` / `refresh_models`；重复注册保留未定义字段
- `register_native_provider(provider: Provider)`：直接注册 `pi_ai.Provider` 实例（覆盖内置同 id）
- `unregister_provider(provider_id)`
- 模型组合：`_compose_models`（内置 > `models` JSON > 扩展 > `modelOverrides`）；`_compose_api_key_auth`（存储凭证 > 配置 key > 内置环境变量）
- `check_auth(provider_id)` / `get_auth(model)`：供主请求、摘要请求复用

## 认证与 OAuth

- API Key：`pi_ai.auth.EnvApiKeyAuth`、`InMemoryCredentialStore`、`resolve_api_key`；环境变量名约定 `{PROVIDER}_API_KEY`
- OAuth：`src/pi_ai/auth/oauth/`（内置 flow，如 GitHub Copilot），`pi_ai.cli login <provider>`；`ModelRuntime` 的 `_resolve_auth` 负责 OAuth 按需刷新与最小有效期（`min_oauth_validity_ms`）
- 无认证的本地服务（如 Ollama）：`Provider(auth=None)`，请求时填充占位 key

## 自定义流实现

流函数签名：`async def my_stream(model, context, options) -> AssistantMessageEventStream`（见 `src/pi_ai/api/completions.py` / `responses.py` 参考实现）。options 支持 `api_key`、`base_url`、`headers`、`env`、`max_tokens`、`reasoning`、`signal`、`cache_retention`、`session_id` 等。流内部按事件推送（text_delta / thinking_delta / toolCall 解析等），以 `done` / `error` 结束；`AssistantMessageEventStream.result()` 返回最终 `AssistantMessage`（含 `stop_reason`、`usage`、`content` 块）。可复用 `pi_ai.utils._event_stream` / `proxy.py` 的事件解析逻辑。

## 配置值解析

`src/pi_coding_agent/resolve_config_value.py`：

- `resolve_config_value_or_throw`：支持 `${ENV_VAR}` 引用与命令求值
- `resolve_headers_or_throw`：headers 里的值同样可引用环境变量
- `get_config_value_env_var_names`：提取依赖的环境变量名（用于可用性快照）

## 测试

- `pi_ai` 单测：`test_provider.py`、`test_providers.py`、`test_api_provider_registry.py`、`test_provider_env.py`、`test_oauth_flows.py`
- 应用层：`src/pi_coding_agent/tests/test_session_model_switch.py` 等
- 参考示例：TS 仓库 `examples/extensions/custom-provider-anthropic/`、`custom-provider-gitlab-duo/`（pi-python 的 Python 化示例见 `examples/extensions/` 下对应 `.py` 文件）
