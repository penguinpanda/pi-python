"""_stream_fn.py 模块测试。"""

import pytest
from pi_agent._stream_fn import get_default_stream_fn, set_default_stream_fn


class TestSetDefaultStreamFn:
    def test_set_and_get(self):
        """设置后可以获取。"""
        called: list = []

        async def mock_fn(model, context, options):
            called.append(True)
            from pi_ai.utils._event_stream import AssistantMessageEventStream

            stream = AssistantMessageEventStream()
            return stream

        set_default_stream_fn(mock_fn)
        fn = get_default_stream_fn()
        assert fn is mock_fn

    def test_get_unset_raises(self):
        """未设置时抛 RuntimeError。"""
        set_default_stream_fn(None)
        with pytest.raises(RuntimeError, match="No default stream function"):
            get_default_stream_fn()

    def test_set_none_clears(self):
        """传 None 可清除。"""
        set_default_stream_fn(None)
        with pytest.raises(RuntimeError):
            get_default_stream_fn()
