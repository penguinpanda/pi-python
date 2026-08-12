"""GitHub Copilot 动态头测试（对齐 TS github-copilot-headers.ts）。"""

from __future__ import annotations

from pi_ai.api.github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
    infer_copilot_initiator,
)


def _user(content):
    return {"role": "user", "content": content, "timestamp": 1}


def _assistant():
    return {"role": "assistant", "content": [], "timestamp": 2}


def test_infer_initiator() -> None:
    assert infer_copilot_initiator([_user("hi")]) == "user"
    assert infer_copilot_initiator([_user("hi"), _assistant()]) == "agent"


def test_has_copilot_vision_input() -> None:
    assert not has_copilot_vision_input([_user("hi")])
    assert has_copilot_vision_input(
        [_user([{"type": "image", "url": None, "data": "xx", "mime_type": "image/png"}])]
    )
    assert has_copilot_vision_input(
        [
            {
                "role": "toolResult",
                "tool_call_id": "t1",
                "tool_name": "read",
                "content": [{"type": "image", "url": None, "data": "xx", "mime_type": "image/png"}],
                "is_error": False,
                "timestamp": 1,
            }
        ]
    )


def test_build_dynamic_headers() -> None:
    headers = build_copilot_dynamic_headers(messages=[_user("hi")], has_images=False)
    assert headers == {
        "X-Initiator": "user",
        "Openai-Intent": "conversation-edits",
    }
    with_images = build_copilot_dynamic_headers(
        messages=[_user("hi"), _assistant()], has_images=True
    )
    assert with_images["X-Initiator"] == "agent"
    assert with_images["Copilot-Vision-Request"] == "true"
