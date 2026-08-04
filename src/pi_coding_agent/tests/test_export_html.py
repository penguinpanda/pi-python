"""HTML 会话导出测试。"""

from __future__ import annotations

import base64
import json
import re

from pi_ai._types import TextContent, UserMessage

from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.export_html import export_session_to_html


def _make_session(tmp_path):
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    import asyncio

    asyncio.run(manager.append_message(UserMessage(role="user", content="hello")))
    asyncio.run(
        manager.append_message(
            {
                "role": "assistant",
                "content": [TextContent(type="text", text="```python\nprint('hi')\n```")],
                "api": "openai-completions",
                "provider": "faux",
                "model": "faux-1",
            }
        )
    )
    return manager


class TestExportHtml:
    def test_exports_file(self, tmp_path):
        manager = _make_session(tmp_path)
        output = tmp_path / "session.html"
        export_session_to_html(manager, output)
        content = output.read_text(encoding="utf-8")
        assert "<html" in content
        assert "User" in content
        assert "Assistant" in content
        assert "hello" in content
        # Pygments 代码高亮。
        assert 'class="code-block"' in content
        assert "print" in content

    def test_embeds_base64_data(self, tmp_path):
        manager = _make_session(tmp_path)
        output = tmp_path / "session.html"
        export_session_to_html(manager, output)
        content = output.read_text(encoding="utf-8")
        match = re.search(r'id="session-data" type="text/plain">([^<]+)<', content)
        assert match is not None
        decoded = json.loads(base64.b64decode(match.group(1)).decode("utf-8"))
        assert decoded["sessionId"] == manager.session_id
        assert decoded["leafId"] == manager.get_leaf_id()
        assert len(decoded["entries"]) == 2

    def test_uses_theme_colors(self, tmp_path):
        from pi_tui.theme import ThemeLoader

        manager = _make_session(tmp_path)
        theme = ThemeLoader().load("dark")
        output = tmp_path / "styled.html"
        export_session_to_html(manager, output, theme=theme)
        content = output.read_text(encoding="utf-8")
        assert theme.colors["bg"] in content
        assert theme.colors["accent"] in content
