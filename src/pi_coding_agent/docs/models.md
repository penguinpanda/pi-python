# 模型（Python 移植）

模型系统分两层：底层 `pi_ai` 的注册表 / 调度，应用层 `pi_coding_agent.ModelRuntime` 的组合与认证。

## `pi_ai.Models`（SDK 统一入口）

`src/pi_ai/models/__init__.py`：

- `add_provider(provider)` 注册 Provider
- `get_model(provider_id, model_id)` / `get_models(provider_id=None)` 查询
- `stream(model, context, options)` / `complete(...)` 请求调度
- `set_api_key(provider_id, api_key)` 凭证管理
- `refresh(options)`：`ModelsRefreshOptions(allow_network=True, force=False, signal=None)`，返回 `ModelsRefreshResult(aborted, errors)`（每个 provider 的错误不抛给调用方）

调度链：`Models.stream → Provider.stream → 按 model.api 查 API 注册表（openai-completions / openai-responses / pi-messages）→ HTTP`。认证在 Provider 层解析（存储凭证 > 配置 key > 环境变量），见 `src/pi_ai/provider.py`。

## `Model` 数据结构

`src/pi_ai/types/model.py`：

- `id`、`provider`、`api`（API 类型）、`name`
- `input` / `output` 能力列表、`max_tokens`（默认 4096）、`context_window`（0=未知）
- `base_url`（模型级覆盖）、`headers`（模型级自定义头）
- `cost`（`ModelCost` + `ModelCostTier` 档位）
- `compat`（OpenRouter / Vercel 网关等兼容配置）
- `thinking_level_map`（pi 思考级别 → provider 值映射）、`reasoning`（是否支持 Thinking）、`deprecated`

## 内置模型目录

- 生成数据：`src/pi_ai/models/generated/providers/*.json`（openai、deepseek、mistral、openrouter 等）
- 目录展开：`flatten_model_catalog`（provider → api → {modelId: Model} 压平，`src/pi_ai/models/model_catalog.py`）
- 存储：`ModelsStore` / `InMemoryModelsStore` / `provider_models_store`（`models_store.py`），刷新带 HTTP 缓存（`http_cache.py`）
- 重新生成目录：`src/pi_ai/scripts/generate_models.py`（**不要手改** `generated` 下的文件）

## 应用层：`ModelRuntime` 组合

`src/pi_coding_agent/model_runtime.py` 把四层来源组合成一个 provider 的最终视图：

1. 内置 provider（`_builtins`）
2. settings 的 `models`（JSON 模型定义，`_apply_models_json`）
3. settings 的 `modelOverrides`（`_apply_model_override`）
4. 扩展注册（`register_provider(provider_id, config)` / `register_native_provider(provider)`，`_apply_extension`）

自定义模型定义支持：`id`、`name`、`api`、`base_url`（必填）、`max_tokens`（默认 16384）、`context_window`（默认 128000）、`headers`、`cost`、`compat`、`thinking_level_map`、`reasoning`。`base_url` 可以在 provider 级或模型级设置；完全没有 base_url 时抛错。

认证组合优先级（`ComposedApiKeyAuth`）：**存储凭证 > models.json / 扩展配置的 apiKey > 内置环境变量**。配置里的 key 支持环境变量引用与命令解析（`resolve_config_value_or_throw`）。

对外 API：`get_models(provider_id)`、`get_model(...)`、`check_auth(provider_id)`、`get_auth(model)`（供摘要请求等复用）。CLI 选择入口：`--model` / `--provider` / `--models`（循环列表）/ `--list-models`。

## 测试

- `src/pi_ai/tests/`：`test_models.py`、`test_model_catalog.py`、`test_models_store.py`、`test_generate_models.py`、`test_provider.py`、`test_api_provider_registry.py`
- 离线假 provider：`src/pi_ai/providers/faux.py`；`pi_evals` harness 默认用它
