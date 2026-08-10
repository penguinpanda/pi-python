"""Phase 4 环境抽象 + 内置工具测试。"""

from __future__ import annotations

import asyncio
import os

import pytest

from pi_agent import (
    BashToolOptions,
    ExecutionError,
    FileError,
    PythonExecutionEnv,
    ShellExecOptions,
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
    get_or_throw,
)


async def _shell_available(env: PythonExecutionEnv) -> bool:
    return (await env._resolve_shell())[0]


def _tool_context(env):
    class _Context:
        pass

    context = _Context()
    context.env = env
    return context


class TestExecutionEnv:
    @pytest.mark.asyncio
    async def test_write_read_roundtrip(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        result = await env.write_file("dir/nested/file.txt", "hello world")
        assert result[0] is True

        read = await env.read_text_file("dir/nested/file.txt")
        assert read == (True, "hello world")

        exists = await env.exists("dir/nested/file.txt")
        assert exists == (True, True)

    @pytest.mark.asyncio
    async def test_not_found_error(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        result = await env.read_text_file("missing.txt")
        assert result[0] is False
        assert isinstance(result[1], FileError)
        assert result[1].code == "not_found"

    @pytest.mark.asyncio
    async def test_list_dir_and_file_info(self, tmp_path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        env = PythonExecutionEnv(str(tmp_path))

        entries = get_or_throw(await env.list_dir("."))
        names = {entry.name for entry in entries}
        assert {"a.txt", "sub"} <= names

        info = get_or_throw(await env.file_info("a.txt"))
        assert info.kind == "file"
        assert info.size == 1

    @pytest.mark.asyncio
    async def test_exec_echo(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        result = await env.exec("echo hello")
        assert result[0] is True
        assert result[1].stdout.strip() == "hello"
        assert result[1].exit_code == 0

    @pytest.mark.asyncio
    async def test_exec_timeout(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        command = "ping -n 10 127.0.0.1 > nul" if os.name == "nt" else "sleep 10"
        result = await env.exec(
            command,
            ShellExecOptions(timeout=0.2),
        )
        assert result[0] is False
        assert isinstance(result[1], ExecutionError)
        assert result[1].code == "timeout"

    @pytest.mark.asyncio
    async def test_remove_and_create_dir(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        await env.write_file("x/y.txt", "content")
        assert (await env.exists("x/y.txt")) == (True, True)
        await env.remove("x/y.txt")
        assert (await env.exists("x/y.txt")) == (True, False)
        await env.create_dir("newdir")
        info = get_or_throw(await env.file_info("newdir"))
        assert info.kind == "directory"

    @pytest.mark.asyncio
    async def test_absolute_and_join_and_canonical(self, tmp_path, monkeypatch):
        env = PythonExecutionEnv(str(tmp_path))
        assert get_or_throw(await env.absolute_path("a/b")) == str(tmp_path / "a" / "b")
        assert get_or_throw(await env.join_path(["a", "b", "c"])) == os.path.join("a", "b", "c")
        target = tmp_path / "real.txt"
        target.write_text("x", encoding="utf-8")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlink not supported")
        assert get_or_throw(await env.canonical_path(str(link))) == str(target.resolve())
        assert (await env.canonical_path(str(tmp_path / "missing")))[0] is True

    @pytest.mark.asyncio
    async def test_abort_signal_short_circuits(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        signal = asyncio.Event()
        signal.set()
        ok, err = await env.read_text_file("x.txt", signal)
        assert ok is False
        assert err.code == "aborted"
        assert (await env.create_temp_dir(signal=signal))[1].code == "aborted"

    @pytest.mark.asyncio
    async def test_read_text_lines_max_lines_and_crlf(self, tmp_path):
        path = tmp_path / "lines.txt"
        path.write_bytes(b"a\r\nb\r\nc\r\n")
        env = PythonExecutionEnv(str(tmp_path))
        lines = get_or_throw(await env.read_text_lines(str(path)))
        assert lines == ["a", "b", "c"]
        assert get_or_throw(await env.read_text_lines(str(path), {"maxLines": 2})) == ["a", "b"]
        assert get_or_throw(await env.read_text_lines(str(path), {"maxLines": 0})) == []
        raw = get_or_throw(await env.read_text_file(str(path)))
        assert raw == "a\r\nb\r\nc\r\n"

    @pytest.mark.asyncio
    async def test_append_and_binary_write(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        await env.append_file("a.txt", "one\n")
        await env.append_file("a.txt", "two\n")
        assert get_or_throw(await env.read_text_file("a.txt")) == "one\ntwo\n"
        await env.write_file("b.bin", b"\x00\x01")
        assert (tmp_path / "b.bin").read_bytes() == b"\x00\x01"

    @pytest.mark.asyncio
    async def test_file_info_symlink(self, tmp_path):
        target = tmp_path / "t.txt"
        target.write_text("x", encoding="utf-8")
        link = tmp_path / "l.txt"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlink not supported")
        env = PythonExecutionEnv(str(tmp_path))
        info = get_or_throw(await env.file_info(str(link)))
        assert info.kind == "symlink"

    @pytest.mark.asyncio
    async def test_create_temp_and_remove_force(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        temp_dir = get_or_throw(await env.create_temp_dir(prefix="pi-t-"))
        assert os.path.isdir(temp_dir)
        temp_file = get_or_throw(await env.create_temp_file({"prefix": "pi-f-", "suffix": ".log"}))
        assert temp_file.endswith(".log")
        await env.remove(temp_file, {"force": True})
        assert os.path.exists(temp_file) is False
        assert (await env.remove(str(tmp_path / "missing"), {"force": True}))[0] is True

    @pytest.mark.asyncio
    async def test_custom_shell_resolution(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path), shell_path="/nonexistent/shell")
        ok, err = await env._resolve_shell()
        assert ok is False
        assert err.code == "shell_unavailable"

    @pytest.mark.asyncio
    async def test_exec_missing_cwd(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        ok, err = await env.exec("echo hi", ShellExecOptions(cwd=str(tmp_path / "nope")))
        assert ok is False
        assert err.code == "spawn_error"

    @pytest.mark.asyncio
    async def test_exec_callback_error_returned(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")

        def bad_callback(_chunk: str) -> None:
            raise RuntimeError("callback exploded")

        ok, err = await env.exec("echo hi", ShellExecOptions(on_stdout=bad_callback))
        assert ok is False
        assert isinstance(err, ExecutionError)
        assert "callback exploded" in str(err)

    @pytest.mark.asyncio
    async def test_exec_abort_kills_process(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        signal = asyncio.Event()
        task = asyncio.create_task(env.exec("sleep 30", ShellExecOptions(abort_signal=signal)))
        await asyncio.sleep(0.2)
        signal.set()
        ok, err = await asyncio.wait_for(task, timeout=10)
        assert ok is False
        assert err.code == "aborted"


class TestReadTool:
    @pytest.mark.asyncio
    async def test_read_text(self, tmp_path):
        (tmp_path / "file.txt").write_bytes(b"line1\nline2\nline3")
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_read_tool()
        result = await tool.execute("t1", {"path": "file.txt"}, None, None, _tool_context(env))
        assert result.content[0]["text"] == "line1\nline2\nline3"

    @pytest.mark.asyncio
    async def test_read_with_offset_limit(self, tmp_path):
        (tmp_path / "file.txt").write_text(
            "\n".join(f"line{i}" for i in range(10)), encoding="utf-8"
        )
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_read_tool()
        result = await tool.execute(
            "t1", {"path": "file.txt", "offset": 2, "limit": 2}, None, None, _tool_context(env)
        )
        text = result.content[0]["text"]
        assert "line1" in text and "line2" in text
        assert "more lines" in text

    @pytest.mark.asyncio
    async def test_read_binary_file_returns_text(self, tmp_path):
        (tmp_path / "bin.dat").write_bytes(bytes([0x00, 0x01, 0x02]))
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_read_tool()
        result = await tool.execute("t1", {"path": "bin.dat"}, None, None, _tool_context(env))
        # 非图片二进制按文本读取（errors=replace）
        assert result.content[0]["text"] is not None

    @pytest.mark.asyncio
    async def test_read_not_found_includes_no_disk_search_guidance(self, tmp_path):
        """回归（P19）：read 找不到文件时提示不要全盘搜索。"""
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_read_tool()
        with pytest.raises(ValueError) as excinfo:
            await tool.execute("t1", {"path": "missing.txt"}, None, None, _tool_context(env))
        assert "do not search the whole disk" in str(excinfo.value)

    def test_read_description_scoped_to_working_directory(self):
        tool = create_read_tool()
        assert "current working directory" in tool.description
        assert "do not search the whole disk" in tool.description

    @pytest.mark.asyncio
    async def test_read_offset_beyond_end_raises(self, tmp_path):
        (tmp_path / "f.txt").write_text("a\nb\n", encoding="utf-8")
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_read_tool()
        with pytest.raises(ValueError, match="beyond end of file"):
            await tool.execute(
                "t1", {"path": "f.txt", "offset": 99}, None, None, _tool_context(env)
            )


class TestWriteTool:
    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_write_tool()
        result = await tool.execute(
            "t1", {"path": "a/b/c.txt", "content": "data"}, None, None, _tool_context(env)
        )
        assert "Successfully wrote 4 bytes" in result.content[0]["text"]
        assert (tmp_path / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "data"


class TestEditTool:
    @pytest.mark.asyncio
    async def test_edit_exact_replacement(self, tmp_path):
        (tmp_path / "code.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_edit_tool()
        result = await tool.execute(
            "t1",
            {"path": "code.py", "edits": [{"oldText": "x = 1", "newText": "x = 10"}]},
            None,
            None,
            _tool_context(env),
        )
        assert "Successfully replaced 1 block" in result.content[0]["text"]
        assert "x = 10" in result.details["patch"]
        assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 10\ny = 2\n"

    @pytest.mark.asyncio
    async def test_edit_preserves_crlf(self, tmp_path):
        (tmp_path / "win.txt").write_bytes(b"a\r\nb\r\nc\r\n")
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_edit_tool()
        await tool.execute(
            "t1",
            {"path": "win.txt", "edits": [{"oldText": "b", "newText": "B"}]},
            None,
            None,
            _tool_context(env),
        )
        assert (tmp_path / "win.txt").read_bytes() == b"a\r\nB\r\nc\r\n"

    @pytest.mark.asyncio
    async def test_edit_fuzzy_unicode_match(self, tmp_path):
        # 智能引号 → ASCII 引号的模糊匹配
        (tmp_path / "f.txt").write_text('say "hello"', encoding="utf-8")
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_edit_tool()
        await tool.execute(
            "t1",
            {
                "path": "f.txt",
                "edits": [{"oldText": "say \u201chello\u201d", "newText": 'say "hi"'}],
            },
            None,
            None,
            _tool_context(env),
        )
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == 'say "hi"'

    @pytest.mark.asyncio
    async def test_edit_missing_text_raises(self, tmp_path):
        (tmp_path / "m.txt").write_text("abc", encoding="utf-8")
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_edit_tool()
        with pytest.raises(ValueError, match="Could not find"):
            await tool.execute(
                "t1",
                {"path": "m.txt", "edits": [{"oldText": "zzz", "newText": "x"}]},
                None,
                None,
                _tool_context(env),
            )

    @pytest.mark.asyncio
    async def test_edit_overlapping_edits_raise(self, tmp_path):
        (tmp_path / "o.txt").write_text("abcdef", encoding="utf-8")
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_edit_tool()
        with pytest.raises(ValueError, match="overlap"):
            await tool.execute(
                "t1",
                {
                    "path": "o.txt",
                    "edits": [
                        {"oldText": "abc", "newText": "ABC"},
                        {"oldText": "bcd", "newText": "BCD"},
                    ],
                },
                None,
                None,
                _tool_context(env),
            )


class TestBashTool:
    def test_bash_description_scoped_to_working_directory(self):
        tool = create_bash_tool()
        assert "current working directory" in tool.description
        assert "do not scan the whole disk" in tool.description

    @pytest.mark.asyncio
    async def test_bash_echo(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        tool = create_bash_tool()
        result = await tool.execute("t1", {"command": "echo hello"}, None, None, _tool_context(env))
        assert result.content[0]["text"].strip() == "hello"

    @pytest.mark.asyncio
    async def test_bash_nonzero_exit_raises(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        tool = create_bash_tool()
        with pytest.raises(ValueError, match="exited with code"):
            await tool.execute("t1", {"command": "exit 3"}, None, None, _tool_context(env))

    @pytest.mark.asyncio
    async def test_bash_timeout_raises(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        tool = create_bash_tool()
        command = "ping -n 10 127.0.0.1 > nul" if os.name == "nt" else "sleep 10"
        with pytest.raises(ValueError, match="timed out"):
            await tool.execute(
                "t1", {"command": command, "timeout": 0.2}, None, None, _tool_context(env)
            )

    @pytest.mark.asyncio
    async def test_bash_abort_raises(self, tmp_path):
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        tool = create_bash_tool()
        signal = asyncio.Event()

        async def _abort() -> None:
            await asyncio.sleep(0.2)
            signal.set()

        task = asyncio.create_task(
            tool.execute("t1", {"command": "sleep 30"}, signal, None, _tool_context(env))
        )
        await asyncio.create_task(_abort())
        with pytest.raises(ValueError, match="aborted"):
            await asyncio.wait_for(task, timeout=10)


class TestFileMutationQueue:
    @pytest.mark.asyncio
    async def test_serialized_writes(self, tmp_path):
        from pi_agent.tools.file_mutation_queue import with_file_mutation_queue

        env = PythonExecutionEnv(str(tmp_path))
        order: list[str] = []

        async def _first() -> None:
            order.append("first-start")
            await asyncio.sleep(0.05)
            order.append("first-end")

        async def _second() -> None:
            order.append("second-start")
            order.append("second-end")

        await asyncio.gather(
            with_file_mutation_queue(env, "same.txt", _first),
            with_file_mutation_queue(env, "same.txt", _second),
        )
        assert order == ["first-start", "first-end", "second-start", "second-end"]


class TestPromptGuidelines:
    def test_read_write_edit_guidelines(self):
        assert create_read_tool().prompt_guidelines == [
            "Use read to examine files instead of cat or sed."
        ]
        assert create_write_tool().prompt_guidelines == [
            "Use write only for new files or complete rewrites."
        ]
        assert create_edit_tool().prompt_guidelines == [
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
        ]

    def test_bash_guideline_conditional(self):
        assert create_bash_tool().prompt_guidelines == [
            "Inspect PI_* environment variables for current model and session details."
        ]
        assert (
            create_bash_tool(BashToolOptions(expose_session_environment=False)).prompt_guidelines
            is None
        )


class TestSessionEnvInjection:
    @pytest.mark.asyncio
    async def test_exec_unset_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_SESSION_ID", "stale")
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        result = await env.exec(
            "python -c \"import os;print(os.environ.get('PI_SESSION_ID',''))\"",
            ShellExecOptions(unset_env=["PI_SESSION_ID"]),
        )
        assert result[0] is True
        assert result[1].stdout.strip() == ""

    @pytest.mark.asyncio
    async def test_bash_tool_deletes_stale_session_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_SESSION_ID", "stale")
        monkeypatch.setenv("PI_PROVIDER", "stale-provider")
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        command = (
            "python -c \"import os;print(os.environ.get('PI_SESSION_ID','')"
            "+'|'+os.environ.get('PI_PROVIDER',''))\""
        )
        tool = create_bash_tool(BashToolOptions(expose_session_environment=False))
        result = await tool.execute("t1", {"command": command}, None, None, _tool_context(env))
        assert result.content[0]["text"].strip() == "|"

    @pytest.mark.asyncio
    async def test_bash_tool_reinjects_session_env_after_delete(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PI_SESSION_ID", "stale")
        monkeypatch.setenv("PI_PROVIDER", "stale-provider")
        env = PythonExecutionEnv(str(tmp_path))
        if not await _shell_available(env):
            pytest.skip("No bash shell available")
        command = (
            "python -c \"import os;print(os.environ.get('PI_SESSION_ID','')"
            "+'|'+os.environ.get('PI_PROVIDER',''))\""
        )
        tool = create_bash_tool(
            BashToolOptions(
                session_env_provider=lambda: {"PI_SESSION_ID": "fresh"},
            )
        )
        result = await tool.execute("t1", {"command": command}, None, None, _tool_context(env))
        assert result.content[0]["text"].strip() == "fresh|"
