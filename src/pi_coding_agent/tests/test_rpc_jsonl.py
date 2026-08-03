"""RPC JSONL 帧编解码测试。"""

from __future__ import annotations

from pi_ai import Model

from pi_coding_agent.rpc.jsonl import serialize_json_line


class TestSerializeJsonLine:
    def test_basic(self):
        assert serialize_json_line({"type": "get_state"}) == '{"type": "get_state"}\n'

    def test_unicode(self):
        line = serialize_json_line({"message": "你好"})
        assert line.endswith("\n")
        assert "你好" in line

    def test_dataclass_default(self):
        model = Model(id="m1", provider="faux", api="openai-completions")
        line = serialize_json_line({"model": model})
        assert '"id": "m1"' in line
        assert '"provider": "faux"' in line

    def test_no_unicode_separator_split(self):
        # U+2028 属于 JSON 字符串合法内容，不应被拆行。
        line = serialize_json_line({"text": "a\u2028b"})
        assert line.count("\n") == 1
        assert "a\u2028b" in line
