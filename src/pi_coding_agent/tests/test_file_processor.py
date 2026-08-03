"""@file 参数处理测试。"""

from __future__ import annotations

from pi_coding_agent.file_processor import process_at_files


class TestProcessAtFiles:
    async def test_plain_args_passthrough(self, tmp_path):
        texts, images = await process_at_files(["hello", "world"], str(tmp_path))
        assert texts == ["hello", "world"]
        assert images == []

    async def test_text_file_injection(self, tmp_path):
        (tmp_path / "note.txt").write_text("file content", encoding="utf-8")
        texts, images = await process_at_files(["@note.txt"], str(tmp_path))
        assert len(texts) == 1
        assert "file content" in texts[0]
        assert "note.txt" in texts[0]
        assert images == []

    async def test_image_file(self, tmp_path):
        (tmp_path / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
        texts, images = await process_at_files(["@pic.png"], str(tmp_path))
        assert texts == []
        assert len(images) == 1
        assert images[0]["mime_type"] == "image/png"
        assert images[0]["data"]

    async def test_directory_recursion(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.md").write_text("doc a", encoding="utf-8")
        (tmp_path / "docs" / "b.txt").write_text("doc b", encoding="utf-8")
        texts, images = await process_at_files(["@docs"], str(tmp_path))
        assert len(texts) == 2
        assert any("doc a" in text for text in texts)
        assert any("doc b" in text for text in texts)

    async def test_missing_path_passthrough(self, tmp_path):
        texts, images = await process_at_files(["@nope.txt"], str(tmp_path))
        assert texts == ["@nope.txt"]
        assert images == []

    async def test_absolute_path(self, tmp_path):
        target = tmp_path / "abs.txt"
        target.write_text("abs content", encoding="utf-8")
        texts, _images = await process_at_files([f"@{target}"], str(tmp_path))
        assert "abs content" in texts[0]
