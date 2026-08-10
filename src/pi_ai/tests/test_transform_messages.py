"""
Unit tests for transform_messages.py — 跨 Provider 消息转换管道。

对齐 TS 测试场景：

    transform-messages-copilot-openai-to-anthropic.test.ts
    lax-message-content.test.ts
"""

from pi_ai._types import (
    Message,
    Model,
)
from pi_ai.api.responses import _to_responses_input
from pi_ai.api.transform_messages import (
    NON_VISION_TOOL_IMAGE_PLACEHOLDER,
    NON_VISION_USER_IMAGE_PLACEHOLDER,
    normalize_responses_tool_call_id,
    normalize_tool_call_id,
    short_hash,
    transform_messages,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_model(
    model_id: str = "test-model",
    provider: str = "test-provider",
    api: str = "openai-completions",
    supports_images: bool = False,
) -> Model:
    return Model(
        id=model_id,
        provider=provider,
        api=api,
        name=model_id,
        input=["text"] + (["image"] if supports_images else []),
        output=["text"],
    )


def _asst(
    content: list[dict] | None,
    provider: str = "src-provider",
    api: str = "openai-responses",
    model: str = "src-model",
    stop_reason: str = "tool_call",
) -> dict:
    """构造一条跨模型 AssistantMessage（默认与 test-model 不同）。"""

    return {
        "role": "assistant",
        "content": content,
        "api": api,
        "provider": provider,
        "model": model,
        "stop_reason": stop_reason,
        "timestamp": 1,
    }


def _tool_call(id_: str, name: str = "bash", arguments: dict | None = None, **extra) -> dict:
    return {
        "type": "toolCall",
        "id": id_,
        "name": name,
        "raw_arguments": "",
        "arguments": arguments if arguments is not None else {},
        **extra,
    }


# ---------------------------------------------------------------------------
# short_hash（对齐 TS utils/hash.ts）
# ---------------------------------------------------------------------------


class TestShortHash:
    def test_matches_ts_reference_values(self):
        # 参考值来自 TS shortHash 实测输出。
        assert short_hash("") == "k4n83c7h0j2b"
        assert short_hash("abc") == "y0biex7f9bbh"
        assert short_hash("call_123|fc_456") == "1l8gxfc1027wt9"

    def test_non_ascii_matches_ts(self):
        # UTF-16 code unit 迭代，非 ASCII 输入与 TS 一致。
        # 参考值来自 TS shortHash 实测输出。
        assert short_hash("你好世界") == "tv7rq2jx1on"
        assert short_hash("emoji: \U0001f600\U0001f680 mixed 中文") == "18h4cz0e802lb"
        assert short_hash("emoji: \U0001f600\U0001f680 mixed 中文 and ascii") == "1sjt4c01p754sm"

    def test_deterministic(self):
        assert short_hash("call_" + "a" * 300) == short_hash("call_" + "a" * 300)


# ---------------------------------------------------------------------------
# 图片降级
# ---------------------------------------------------------------------------


class TestImageDowngrade:
    def test_user_image_downgraded_for_non_vision(self):
        model = _make_model(supports_images=False)
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe:"},
                    {
                        "type": "image",
                        "url": "https://example.com/p.png",
                        "data": None,
                        "mime_type": None,
                    },
                ],
                "timestamp": 1,
            }
        ]
        result = transform_messages(messages, model)
        assert result[0]["content"] == [
            {"type": "text", "text": "Describe:"},
            {"type": "text", "text": NON_VISION_USER_IMAGE_PLACEHOLDER},
        ]

    def test_tool_result_image_downgraded(self):
        model = _make_model(supports_images=False)
        messages: list[Message] = [
            {
                "role": "toolResult",
                "tool_call_id": "call_1",
                "tool_name": "web",
                "content": [
                    {"type": "image", "url": None, "data": "abcd", "mime_type": "image/png"},
                ],
                "is_error": False,
                "timestamp": 1,
            }
        ]
        result = transform_messages(messages, model)
        assert result[0]["content"] == [
            {"type": "text", "text": NON_VISION_TOOL_IMAGE_PLACEHOLDER},
        ]

    def test_consecutive_images_deduplicated(self):
        model = _make_model(supports_images=False)
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "a", "data": None, "mime_type": None},
                    {"type": "image", "url": "b", "data": None, "mime_type": None},
                    {"type": "text", "text": "after"},
                ],
                "timestamp": 1,
            }
        ]
        result = transform_messages(messages, model)
        assert result[0]["content"] == [
            {"type": "text", "text": NON_VISION_USER_IMAGE_PLACEHOLDER},
            {"type": "text", "text": "after"},
        ]

    def test_vision_model_keeps_images(self):
        model = _make_model(supports_images=True)
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "a", "data": None, "mime_type": None},
                ],
                "timestamp": 1,
            }
        ]
        result = transform_messages(messages, model)
        assert result[0]["content"] == messages[0]["content"]

    def test_model_input_none_no_crash(self):
        model = _make_model(supports_images=False)
        model.input = None  # type: ignore[assignment]
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "a", "data": None, "mime_type": None},
                ],
                "timestamp": 1,
            }
        ]
        result = transform_messages(messages, model)
        assert result[0]["content"] == [
            {"type": "text", "text": NON_VISION_USER_IMAGE_PLACEHOLDER},
        ]


# ---------------------------------------------------------------------------
# null content 归一化
# ---------------------------------------------------------------------------


class TestNullContent:
    def test_null_content_normalized_to_empty_list(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": None, "timestamp": 1},  # type: ignore[typeddict-item]
            _asst(None, stop_reason="stop"),  # type: ignore[arg-type]
            {
                "role": "toolResult",
                "tool_call_id": "call_1",
                "tool_name": "web",
                "content": None,  # type: ignore[typeddict-item]
                "is_error": False,
                "timestamp": 1,
            },
        ]
        result = transform_messages(messages, model)
        assert len(result) == 3
        for msg in result:
            assert msg["content"] == []


# ---------------------------------------------------------------------------
# Thinking 块处理
# ---------------------------------------------------------------------------


class TestThinkingBlocks:
    def test_thinking_cross_model_converted_to_text(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "hello", "timestamp": 1},
            _asst(
                [
                    {
                        "type": "thinking",
                        "thinking": "Let me think...",
                        "thinking_signature": "reasoning_content",
                    },
                    {"type": "text", "text": "Hi!"},
                ]
            ),
        ]
        result = transform_messages(messages, model)
        asst = result[1]
        thinking_blocks = [b for b in asst["content"] if b["type"] == "thinking"]
        text_blocks = [b for b in asst["content"] if b["type"] == "text"]
        assert thinking_blocks == []
        assert text_blocks == [
            {"type": "text", "text": "Let me think..."},
            {"type": "text", "text": "Hi!"},
        ]

    def test_thinking_redacted_dropped_cross_model(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            _asst(
                [
                    {
                        "type": "thinking",
                        "thinking": "",
                        "thinking_signature": "sig",
                        "redacted": True,
                    },
                    {"type": "text", "text": "answer"},
                ]
            ),
        ]
        result = transform_messages(messages, model)
        assert result[1]["content"] == [{"type": "text", "text": "answer"}]

    def test_thinking_redacted_kept_same_model(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            _asst(
                [
                    {
                        "type": "thinking",
                        "thinking": "",
                        "thinking_signature": "sig",
                        "redacted": True,
                    },
                    {"type": "text", "text": "answer"},
                ],
                provider=model.provider,
                api=model.api,
                model=model.id,
            ),
        ]
        result = transform_messages(messages, model)
        assert result[1]["content"][0]["type"] == "thinking"

    def test_thinking_empty_dropped(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            _asst(
                [
                    {"type": "thinking", "thinking": "   "},
                    {"type": "text", "text": "answer"},
                ]
            ),
        ]
        result = transform_messages(messages, model)
        assert result[1]["content"] == [{"type": "text", "text": "answer"}]

    def test_thinking_signature_kept_same_model_even_empty(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            _asst(
                [
                    {"type": "thinking", "thinking": "", "thinking_signature": "rs_123"},
                    {"type": "text", "text": "answer"},
                ],
                provider=model.provider,
                api=model.api,
                model=model.id,
            ),
        ]
        result = transform_messages(messages, model)
        assert result[1]["content"][0] == {
            "type": "thinking",
            "thinking": "",
            "thinking_signature": "rs_123",
        }


# ---------------------------------------------------------------------------
# Text 块：跨模型剥 text_signature
# ---------------------------------------------------------------------------


class TestTextBlocks:
    def test_text_signature_stripped_cross_model(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            _asst(
                [
                    {"type": "text", "text": "hello", "text_signature": "msg_123"},
                ]
            ),
        ]
        result = transform_messages(messages, model)
        assert result[1]["content"] == [{"type": "text", "text": "hello"}]

    def test_text_signature_kept_same_model(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            _asst(
                [
                    {"type": "text", "text": "hello", "text_signature": "msg_123"},
                ],
                provider=model.provider,
                api=model.api,
                model=model.id,
            ),
        ]
        result = transform_messages(messages, model)
        assert result[1]["content"] == [
            {"type": "text", "text": "hello", "text_signature": "msg_123"},
        ]

    def test_duplicate_identical_text_blocks_deduplicated(self):
        """旧会话中 toolCall 前后重复的 text 块应去重，避免拆散工具调用配对。"""
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            _asst(
                [
                    {"type": "text", "text": "same"},
                    _tool_call("call_1"),
                    {"type": "text", "text": "same"},
                ],
                provider=model.provider,
                api=model.api,
                model=model.id,
            ),
        ]
        result = transform_messages(messages, model)
        assert result[1]["content"] == [
            {"type": "text", "text": "same"},
            _tool_call("call_1"),
        ]


# ---------------------------------------------------------------------------
# Tool Call ID 规范化 + thought_signature
# ---------------------------------------------------------------------------


class TestToolCallNormalization:
    def test_tool_call_id_normalized_and_result_remapped(self):
        model = _make_model(model_id="gpt-5", provider="openai", api="openai-completions")
        raw_id = "call_1|fc_abcdefghijklmnopqrstuvwxyz1234567890"
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst([_tool_call(raw_id, "bash", {"command": "ls"})]),
            {
                "role": "toolResult",
                "tool_call_id": raw_id,
                "tool_name": "bash",
                "content": [{"type": "text", "text": "out"}],
                "is_error": False,
                "timestamp": 1,
            },
        ]
        result = transform_messages(messages, model, normalize_tool_call_id)
        asst = result[1]
        tr = result[2]
        assert "|" not in tr["tool_call_id"]
        assert tr["tool_call_id"].startswith("call_1_")
        # toolResult 的 tool_call_id 与 assistant 的 toolCall id 一致（重映射成功）。
        assert asst["content"][0]["id"] == tr["tool_call_id"]

    def test_short_pipe_id_kept_readable(self):
        model = _make_model(model_id="gpt-5", provider="openai", api="openai-completions")
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst([_tool_call("call_1|fc_123", "bash", {})]),
            {
                "role": "toolResult",
                "tool_call_id": "call_1|fc_123",
                "tool_name": "bash",
                "content": [{"type": "text", "text": "out"}],
                "is_error": False,
                "timestamp": 1,
            },
        ]
        result = transform_messages(messages, model, normalize_tool_call_id)
        assert result[2]["tool_call_id"] == "call_1_fc_123"

    def test_thought_signature_removed_cross_model(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst(
                [
                    _tool_call(
                        "call_123",
                        "bash",
                        {"command": "ls"},
                        thought_signature='{"type":"reasoning.encrypted"}',
                    ),
                ]
            ),
        ]
        result = transform_messages(messages, model)
        assert "thought_signature" not in result[1]["content"][0]

    def test_thought_signature_kept_same_model(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst(
                [_tool_call("call_123", "bash", {"command": "ls"}, thought_signature="sig")],
                provider=model.provider,
                api=model.api,
                model=model.id,
            ),
        ]
        result = transform_messages(messages, model)
        assert result[1]["content"][0]["thought_signature"] == "sig"


# ---------------------------------------------------------------------------
# Responses 系 Tool Call ID 规范化（fc_ item id）
# ---------------------------------------------------------------------------


class TestResponsesToolCallNormalization:
    def test_pipe_id_normalized_and_result_remapped(self):
        model = _make_model(model_id="gpt-5", provider="openai", api="openai-responses")
        raw_id = "call_1|fc_abc"
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst(
                [_tool_call(raw_id, "bash", {})],
                provider="openai",
                api="openai-responses",
                model="gpt-5",
            ),
            {
                "role": "toolResult",
                "tool_call_id": raw_id,
                "tool_name": "bash",
                "content": [{"type": "text", "text": "out"}],
                "is_error": False,
                "timestamp": 1,
            },
        ]
        result = transform_messages(messages, model, normalize_responses_tool_call_id)
        asst = result[1]
        tr = result[2]
        # 同模型：双段 ID 原样保留，toolResult 重映射一致。
        assert asst["content"][0]["id"] == "call_1|fc_abc"
        assert tr["tool_call_id"] == "call_1|fc_abc"

    def test_foreign_pipe_id_hashed_to_fc(self):
        model = _make_model(model_id="gpt-5", provider="openai", api="openai-responses")
        raw_id = "call_1|fc_xyz"
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            # 默认 _asst 是 src-provider → 跨模型。
            _asst([_tool_call(raw_id, "bash", {})]),
            {
                "role": "toolResult",
                "tool_call_id": raw_id,
                "tool_name": "bash",
                "content": [{"type": "text", "text": "out"}],
                "is_error": False,
                "timestamp": 1,
            },
        ]
        result = transform_messages(messages, model, normalize_responses_tool_call_id)
        asst = result[1]
        tr = result[2]
        # 跨模型：item id 用 short_hash 重建为 fc_ 短 id，并保持双段结构。
        assert asst["content"][0]["id"] == tr["tool_call_id"]
        call_id, _, item_id = tr["tool_call_id"].partition("|")
        assert call_id == "call_1"
        assert item_id.startswith("fc_")
        assert len(item_id) <= 64

    def test_disallowed_provider_degrades_to_single_part(self):
        model = _make_model(model_id="qwen-plus", provider="qwen", api="openai-completions")
        raw_id = "call_1|fc_abc"
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst(
                [_tool_call(raw_id, "bash", {})],
                provider="openai",
                api="openai-responses",
                model="gpt-5",
            ),
            {
                "role": "toolResult",
                "tool_call_id": raw_id,
                "tool_name": "bash",
                "content": [{"type": "text", "text": "out"}],
                "is_error": False,
                "timestamp": 1,
            },
        ]
        result = transform_messages(messages, model, normalize_responses_tool_call_id)
        assert result[1]["content"][0]["id"] == "call_1_fc_abc"
        assert result[2]["tool_call_id"] == "call_1_fc_abc"


# ---------------------------------------------------------------------------
# 第二遍：孤立 tool call 合成 + error/aborted 跳过
# ---------------------------------------------------------------------------


class TestSyntheticToolResults:
    def test_synthetic_result_for_trailing_orphaned_tool_calls(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "read file", "timestamp": 1},
            _asst([_tool_call("call_1", "read", {"path": "README.md"})]),
        ]
        result = transform_messages(messages, model)
        last = result[-1]
        assert last["role"] == "toolResult"
        assert last["tool_call_id"] == "call_1"
        assert last["tool_name"] == "read"
        assert last["is_error"] is True
        assert last["content"] == [{"type": "text", "text": "No result provided"}]

    def test_synthetic_only_for_missing_results(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst(
                [
                    _tool_call("call_1", "read", {}),
                    _tool_call("call_2", "bash", {}),
                ]
            ),
            {
                "role": "toolResult",
                "tool_call_id": "call_1",
                "tool_name": "read",
                "content": [{"type": "text", "text": "done"}],
                "is_error": False,
                "timestamp": 1,
            },
        ]
        result = transform_messages(messages, model)
        synthetic = [m for m in result if m["role"] == "toolResult" and m["is_error"]]
        assert len(synthetic) == 1
        assert synthetic[0]["tool_call_id"] == "call_2"
        assert synthetic[0]["tool_name"] == "bash"

    def test_user_message_flushes_pending_tool_calls(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst([_tool_call("call_1", "bash", {})]),
            {"role": "user", "content": "actually stop", "timestamp": 1},
        ]
        result = transform_messages(messages, model)
        assert [m["role"] for m in result] == ["user", "assistant", "toolResult", "user"]
        assert result[2]["tool_call_id"] == "call_1"
        assert result[2]["is_error"] is True

    def test_error_assistant_skipped(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            _asst([{"type": "text", "text": "partial"}], stop_reason="error"),
            _asst([{"type": "text", "text": "good"}], stop_reason="stop"),
        ]
        result = transform_messages(messages, model)
        assert [m["role"] for m in result] == ["user", "assistant"]
        assert result[1]["content"] == [{"type": "text", "text": "good"}]

    def test_aborted_assistant_skipped(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            _asst([{"type": "text", "text": "partial"}], stop_reason="aborted"),
            _asst([{"type": "text", "text": "good"}], stop_reason="stop"),
        ]
        result = transform_messages(messages, model)
        assert [m["role"] for m in result] == ["user", "assistant"]
        assert result[1]["content"] == [{"type": "text", "text": "good"}]


# ---------------------------------------------------------------------------
# AgentMessage 透传（TS 无此类型，Python 刻意保持语义）
# ---------------------------------------------------------------------------


class TestAgentMessagePassthrough:
    def test_unknown_role_passes_through(self):
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "q", "timestamp": 1},
            {"role": "observation", "content": "observed", "name": "tool"},  # type: ignore[typeddict-item]
        ]
        result = transform_messages(messages, model)
        assert len(result) == 2
        assert result[1]["role"] == "observation"
        assert result[1]["content"] == "observed"

    def test_agent_message_does_not_flush_pending_tool_calls(self):
        """observation 等角色不打断工具流（与 TS 仅 user 打断的语义一致）。"""
        model = _make_model()
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst([_tool_call("call_1", "bash", {})]),
            {"role": "observation", "content": "mid", "name": "tool"},  # type: ignore[typeddict-item]
        ]
        result = transform_messages(messages, model)
        # observation 不触发 flush：孤立 tool call 只在结尾统一合成。
        assert [m["role"] for m in result] == ["user", "assistant", "observation", "toolResult"]
        assert result[-1]["tool_call_id"] == "call_1"


# ---------------------------------------------------------------------------
# Responses 侧集成：function_call 历史 + 合成结果合法
# ---------------------------------------------------------------------------


class TestResponsesIntegration:
    def test_responses_function_call_history(self):
        model = _make_model(model_id="gpt-5", provider="openai", api="openai-responses")
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst(
                [_tool_call("call_1", "bash", {"command": "ls"})],
                provider="openai",
                api="openai-responses",
                model="gpt-5",
            ),
            {
                "role": "toolResult",
                "tool_call_id": "call_1",
                "tool_name": "bash",
                "content": [{"type": "text", "text": "out"}],
                "is_error": False,
                "timestamp": 1,
            },
        ]
        transformed = transform_messages(messages, model)
        items = _to_responses_input(transformed, model)
        # 工具调用历史 → 顶层 function_call item。
        fc = next(i for i in items if i.get("type") == "function_call")
        assert fc == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "bash",
            "arguments": '{"command": "ls"}',
        }
        # 对应 toolResult → function_call_output。
        fco = next(i for i in items if i.get("type") == "function_call_output")
        assert fco["call_id"] == "call_1"

    def test_responses_synthetic_result_valid(self):
        model = _make_model(model_id="gpt-5", provider="openai", api="openai-responses")
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst(
                [_tool_call("call_9", "bash", {})],
                provider="openai",
                api="openai-responses",
                model="gpt-5",
            ),
        ]
        transformed = transform_messages(messages, model)
        items = _to_responses_input(transformed, model)
        fco = next(i for i in items if i.get("type") == "function_call_output")
        assert fco["call_id"] == "call_9"
        assert fco["output"] == "No result provided"

    def test_duplicate_text_does_not_split_tool_call_output(self):
        model = _make_model(model_id="gpt-5", provider="openai", api="openai-responses")
        messages: list[Message] = [
            {"role": "user", "content": "run", "timestamp": 1},
            _asst(
                [
                    {"type": "text", "text": "checking"},
                    _tool_call("call_1", "bash", {"command": "ls"}),
                    {"type": "text", "text": "checking"},
                ],
                provider="openai",
                api="openai-responses",
                model="gpt-5",
            ),
            {
                "role": "toolResult",
                "tool_call_id": "call_1",
                "tool_name": "bash",
                "content": [{"type": "text", "text": "out"}],
                "is_error": False,
                "timestamp": 1,
            },
        ]
        transformed = transform_messages(messages, model)
        items = _to_responses_input(transformed, model)
        types = [item.get("type") for item in items]
        fc_index = types.index("function_call")
        assert types[fc_index + 1] == "function_call_output"
