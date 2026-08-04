"""ExecutionEnv 环境抽象（Phase 4.6）。

对齐 TS `harness/env/nodejs.ts` + `types.ts` 的 FileSystem/Shell 契约：

- FileSystem 方法一律返回 `Result[T, FileError]`（不抛异常）
- Shell.exec 返回 `Result[{stdout, stderr, exitCode}, ExecutionError]`
- PythonExecutionEnv 为 Python 平台实现（pathlib + asyncio subprocess）
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Tuple, TypeAlias


# ---------------------------------------------------------------------------
# Result / 错误
# ---------------------------------------------------------------------------

Result: TypeAlias = Tuple[Literal[True], Any] | Tuple[Literal[False], "FileError"]


def ok(value: Any) -> Tuple[Literal[True], Any]:
    return (True, value)


def err(error: "FileError") -> Tuple[Literal[False], "FileError"]:
    return (False, error)


def get_or_throw(result: Result) -> Any:
    """返回成功值或抛出错误（对齐 TS getOrThrow）。"""
    ok_flag, value = result
    if ok_flag:
        return value
    raise value


FileErrorCode = Literal[
    "aborted",
    "not_found",
    "permission_denied",
    "not_directory",
    "is_directory",
    "invalid",
    "not_supported",
    "unknown",
]


class FileError(Exception):
    """后端无关的文件错误。"""

    def __init__(
        self,
        code: FileErrorCode,
        message: str,
        path: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.cause = cause


ExecutionErrorCode = Literal[
    "aborted",
    "timeout",
    "shell_unavailable",
    "spawn_error",
    "callback_error",
    "unknown",
]


class ExecutionError(Exception):
    """后端无关的执行错误。"""

    def __init__(
        self,
        code: ExecutionErrorCode,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


def to_file_error(error: BaseException, path: str | None = None) -> FileError:
    if isinstance(error, FileError):
        return error
    if isinstance(error, FileNotFoundError):
        return FileError("not_found", str(error), path, error)
    if isinstance(error, PermissionError):
        return FileError("permission_denied", str(error), path, error)
    if isinstance(error, NotADirectoryError):
        return FileError("not_directory", str(error), path, error)
    if isinstance(error, IsADirectoryError):
        return FileError("is_directory", str(error), path, error)
    if isinstance(error, ValueError):
        return FileError("invalid", str(error), path, error)
    return FileError("unknown", str(error), path, error)


def to_execution_error(error: BaseException) -> ExecutionError:
    if isinstance(error, ExecutionError):
        return error
    return ExecutionError("unknown", str(error), error)


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------

FileKind = Literal["file", "directory", "symlink"]


class FileInfo:
    """文件系统对象元数据。"""

    def __init__(
        self,
        name: str,
        path: str,
        kind: FileKind,
        size: int,
        mtime_ms: float,
    ) -> None:
        self.name = name
        self.path = path
        self.kind = kind
        self.size = size
        self.mtime_ms = mtime_ms


class ShellExecOptions:
    """Shell.exec 选项。"""

    def __init__(
        self,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        inherit_env: bool = True,
        timeout: float | None = None,
        abort_signal: asyncio.Event | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> None:
        self.cwd = cwd
        self.env = env
        self.inherit_env = inherit_env
        self.timeout = timeout
        self.abort_signal = abort_signal
        self.on_stdout = on_stdout
        self.on_stderr = on_stderr


class ShellResult:
    def __init__(self, stdout: str, stderr: str, exit_code: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


# ---------------------------------------------------------------------------
# 协议
# ---------------------------------------------------------------------------


class FileSystem(Protocol):
    """文件系统能力（所有方法不抛异常，错误编码进 Result）。"""

    cwd: str

    async def absolute_path(self, path: str, signal: asyncio.Event | None = None) -> Result: ...

    async def join_path(self, parts: list[str], signal: asyncio.Event | None = None) -> Result: ...

    async def read_text_file(self, path: str, signal: asyncio.Event | None = None) -> Result: ...

    async def read_text_lines(self, path: str, options: dict[str, Any] | None = None) -> Result: ...

    async def read_binary_file(self, path: str, signal: asyncio.Event | None = None) -> Result: ...

    async def write_file(
        self, path: str, content: str | bytes, signal: asyncio.Event | None = None
    ) -> Result: ...

    async def append_file(
        self, path: str, content: str | bytes, signal: asyncio.Event | None = None
    ) -> Result: ...

    async def file_info(self, path: str, signal: asyncio.Event | None = None) -> Result: ...

    async def list_dir(self, path: str, signal: asyncio.Event | None = None) -> Result: ...

    async def canonical_path(self, path: str, signal: asyncio.Event | None = None) -> Result: ...

    async def exists(self, path: str, signal: asyncio.Event | None = None) -> Result: ...

    async def create_dir(self, path: str, options: dict[str, Any] | None = None) -> Result: ...

    async def remove(self, path: str, options: dict[str, Any] | None = None) -> Result: ...

    async def create_temp_dir(
        self, prefix: str = "tmp-", signal: asyncio.Event | None = None
    ) -> Result: ...

    async def create_temp_file(self, options: dict[str, Any] | None = None) -> Result: ...

    async def cleanup(self) -> None: ...


class Shell(Protocol):
    """Shell 执行能力。"""

    async def exec(self, command: str, options: ShellExecOptions | None = None) -> Result: ...

    async def cleanup(self) -> None: ...


class ExecutionEnv(FileSystem, Shell, Protocol):
    """文件系统 + 进程执行环境。"""


# ---------------------------------------------------------------------------
# Python 平台实现
# ---------------------------------------------------------------------------


class PythonExecutionEnv:
    """Python 平台 ExecutionEnv（pathlib + asyncio subprocess）。"""

    def __init__(
        self,
        cwd: str,
        shell_path: str | None = None,
        shell_env: dict[str, str] | None = None,
    ) -> None:
        self.cwd = os.path.abspath(cwd)
        self._shell_path = shell_path
        self._shell_env = dict(shell_env) if shell_env else None
        self._active_processes: set[asyncio.subprocess.Process] = set()

    # ------------------------------------------------------------------
    # 路径
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> str:
        normalized = path
        home = os.path.expanduser("~")
        if normalized == "~":
            normalized = home
        elif normalized.startswith("~/"):
            normalized = os.path.join(home, normalized[2:])
        elif normalized.startswith("file://"):
            try:
                from urllib.parse import unquote, urlparse

                parsed = urlparse(normalized)
                normalized = unquote(parsed.path)
            except Exception:
                pass
        if os.path.isabs(normalized):
            return os.path.normpath(normalized)
        return os.path.normpath(os.path.join(self.cwd, normalized))

    @staticmethod
    def _aborted(signal: asyncio.Event | None, path: str | None = None) -> Result | None:
        if signal is not None and signal.is_set():
            return (False, FileError("aborted", "aborted", path))
        return None

    async def absolute_path(self, path: str, signal: asyncio.Event | None = None) -> Result:
        aborted = self._aborted(signal, path)
        if aborted:
            return aborted
        return (True, self._resolve_path(path))

    async def join_path(self, parts: list[str], signal: asyncio.Event | None = None) -> Result:
        aborted = self._aborted(signal)
        if aborted:
            return aborted
        return (True, os.path.join(*parts))

    async def canonical_path(self, path: str, signal: asyncio.Event | None = None) -> Result:
        resolved = self._resolve_path(path)
        try:
            return (True, os.path.realpath(resolved))
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    # ------------------------------------------------------------------
    # 文件
    # ------------------------------------------------------------------

    async def read_text_file(self, path: str, signal: asyncio.Event | None = None) -> Result:
        resolved = self._resolve_path(path)
        aborted = self._aborted(signal, resolved)
        if aborted:
            return aborted
        try:
            # Read as bytes first to preserve original line endings (\r\n, \r, \n).
            # Path.read_text() would apply universal newline conversion,
            # losing \r before \n.
            data = await asyncio.to_thread(Path(resolved).read_bytes)
            return (True, data.decode("utf-8"))
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    async def read_text_lines(self, path: str, options: dict[str, Any] | None = None) -> Result:
        resolved = self._resolve_path(path)
        options = options or {}
        signal = options.get("abortSignal")
        aborted = self._aborted(signal, resolved)
        if aborted:
            return aborted
        max_lines = options.get("maxLines")
        if max_lines is not None and max_lines <= 0:
            return (True, [])
        try:
            content = await asyncio.to_thread(Path(resolved).read_text, encoding="utf-8")
            lines = content.split("\n")
            if content.endswith("\n"):
                lines.pop()
            if max_lines is not None and len(lines) > max_lines:
                lines = lines[:max_lines]
            return (True, lines)
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    async def read_binary_file(self, path: str, signal: asyncio.Event | None = None) -> Result:
        resolved = self._resolve_path(path)
        aborted = self._aborted(signal, resolved)
        if aborted:
            return aborted
        try:
            return (True, await asyncio.to_thread(Path(resolved).read_bytes))
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    async def write_file(
        self, path: str, content: str | bytes, signal: asyncio.Event | None = None
    ) -> Result:
        resolved = self._resolve_path(path)
        aborted = self._aborted(signal, resolved)
        if aborted:
            return aborted
        try:
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
            mode = "wb" if isinstance(content, bytes) else "w"
            # read_text_file 按字节读取并保留原始行尾（\r\n / \r / \n），
            # 因此文本写入同样禁用换行翻译（newline=""），避免 Windows 上
            # 把内容里已有的 \r\n 再次翻译成 \r\r\n。
            kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8", "newline": ""}
            await asyncio.to_thread(self._write_plain, resolved, content, mode, kwargs)
            return (True, None)
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    @staticmethod
    def _write_plain(path: str, content: str | bytes, mode: str, kwargs: dict) -> None:
        with open(path, mode, **kwargs) as handle:
            handle.write(content)

    async def append_file(
        self, path: str, content: str | bytes, signal: asyncio.Event | None = None
    ) -> Result:
        resolved = self._resolve_path(path)
        aborted = self._aborted(signal, resolved)
        if aborted:
            return aborted
        try:
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
            mode = "ab" if isinstance(content, bytes) else "a"
            kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8", "newline": ""}
            await asyncio.to_thread(self._write_plain, resolved, content, mode, kwargs)
            return (True, None)
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    async def file_info(self, path: str, signal: asyncio.Event | None = None) -> Result:
        resolved = self._resolve_path(path)
        aborted = self._aborted(signal, resolved)
        if aborted:
            return aborted
        try:
            stats = await asyncio.to_thread(os.lstat, resolved)
            kind: FileKind
            if os.path.islink(resolved):
                kind = "symlink"
            elif os.path.isdir(resolved):
                kind = "directory"
            elif os.path.isfile(resolved):
                kind = "file"
            else:
                return (False, FileError("invalid", "Unsupported file type", resolved))
            return (
                True,
                FileInfo(
                    name=os.path.basename(resolved.rstrip("/\\")) or resolved,
                    path=resolved,
                    kind=kind,
                    size=stats.st_size,
                    mtime_ms=stats.st_mtime * 1000,
                ),
            )
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    async def list_dir(self, path: str, signal: asyncio.Event | None = None) -> Result:
        resolved = self._resolve_path(path)
        aborted = self._aborted(signal, resolved)
        if aborted:
            return aborted
        try:
            names = await asyncio.to_thread(os.listdir, resolved)
            infos: list[FileInfo] = []
            for name in names:
                entry_path = os.path.join(resolved, name)
                try:
                    stats = os.lstat(entry_path)
                    kind: FileKind
                    if os.path.islink(entry_path):
                        kind = "symlink"
                    elif os.path.isdir(entry_path):
                        kind = "directory"
                    elif os.path.isfile(entry_path):
                        kind = "file"
                    else:
                        continue
                    infos.append(
                        FileInfo(
                            name=name,
                            path=entry_path,
                            kind=kind,
                            size=stats.st_size,
                            mtime_ms=stats.st_mtime * 1000,
                        )
                    )
                except BaseException as error:
                    return (False, to_file_error(error, entry_path))
            return (True, infos)
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    async def exists(self, path: str, signal: asyncio.Event | None = None) -> Result:
        info = await self.file_info(path, signal)
        if info[0]:
            return (True, True)
        if info[1].code == "not_found":
            return (True, False)
        return info

    async def create_dir(self, path: str, options: dict[str, Any] | None = None) -> Result:
        resolved = self._resolve_path(path)
        options = options or {}
        try:
            await asyncio.to_thread(
                Path(resolved).mkdir,
                parents=options.get("recursive", True),
                exist_ok=options.get("recursive", True),
            )
            return (True, None)
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    async def remove(self, path: str, options: dict[str, Any] | None = None) -> Result:
        resolved = self._resolve_path(path)
        options = options or {}
        try:
            target = Path(resolved)
            if target.is_dir():
                shutil.rmtree(resolved, ignore_errors=options.get("force", False))
            else:
                await asyncio.to_thread(target.unlink, missing_ok=options.get("force", False))
            return (True, None)
        except BaseException as error:
            return (False, to_file_error(error, resolved))

    async def create_temp_dir(
        self, prefix: str = "tmp-", signal: asyncio.Event | None = None
    ) -> Result:
        aborted = self._aborted(signal)
        if aborted:
            return aborted
        try:
            return (True, tempfile.mkdtemp(prefix=prefix))
        except BaseException as error:
            return (False, to_file_error(error))

    async def create_temp_file(self, options: dict[str, Any] | None = None) -> Result:
        options = options or {}
        try:
            fd, file_path = tempfile.mkstemp(
                prefix=options.get("prefix", ""),
                suffix=options.get("suffix", ""),
            )
            os.close(fd)
            return (True, file_path)
        except BaseException as error:
            return (False, to_file_error(error))

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------

    async def _resolve_shell(self) -> Result:
        if self._shell_path:
            if os.path.exists(self._shell_path):
                return (True, self._shell_path)
            return (
                False,
                ExecutionError(
                    "shell_unavailable",
                    f"Custom shell path not found: {self._shell_path}",
                ),
            )
        if os.name == "nt":
            candidates: list[str] = []
            program_files = os.environ.get("ProgramFiles")
            if program_files:
                candidates.append(os.path.join(program_files, "Git", "bin", "bash.exe"))
            program_files_x86 = os.environ.get("ProgramFiles(x86)")
            if program_files_x86:
                candidates.append(os.path.join(program_files_x86, "Git", "bin", "bash.exe"))
            for candidate in candidates:
                if os.path.exists(candidate):
                    return (True, candidate)
            bash_on_path = shutil.which("bash")
            if bash_on_path:
                return (True, bash_on_path)
            # 无 bash 时回退到 cmd（Windows 自带），保证 bash 工具在未安装
            # Git for Windows 的机器上仍可用。
            cmd_on_path = shutil.which("cmd")
            if cmd_on_path:
                return (True, cmd_on_path)
            return (
                False,
                ExecutionError(
                    "shell_unavailable",
                    "No shell found. Install Git for Windows or configure an explicit shell_path.",
                ),
            )
        if os.path.exists("/bin/bash"):
            return (True, "/bin/bash")
        sh = shutil.which("sh")
        if sh:
            return (True, sh)
        return (False, ExecutionError("shell_unavailable", "No shell found"))

    async def exec(self, command: str, options: ShellExecOptions | None = None) -> Result:
        options = options or ShellExecOptions()
        if options.abort_signal is not None and options.abort_signal.is_set():
            return (False, ExecutionError("aborted", "aborted"))
        timeout = options.timeout
        if timeout is not None and (timeout <= 0 or not isinstance(timeout, (int, float))):
            return (
                False,
                ExecutionError("timeout", "Invalid timeout: must be a finite number of seconds"),
            )

        cwd = self._resolve_path(options.cwd) if options.cwd else self.cwd
        if not os.path.isdir(cwd):
            return (
                False,
                ExecutionError(
                    "spawn_error",
                    f"Working directory does not exist: {cwd}",
                ),
            )

        shell_result = await self._resolve_shell()
        if not shell_result[0]:
            return shell_result
        shell = shell_result[1]

        env = dict(os.environ) if options.inherit_env else {}
        if self._shell_env:
            env.update(self._shell_env)
        if options.env:
            env.update(options.env)

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        timed_out = False
        callback_error: ExecutionError | None = None

        def _on_stdout(chunk: str) -> None:
            stdout_chunks.append(chunk)
            try:
                if options.on_stdout:
                    options.on_stdout(chunk)
            except BaseException as error:
                nonlocal callback_error
                callback_error = to_execution_error(error)
                raise

        creationflags = 0
        start_new_session = os.name != "nt"
        if os.name == "nt":
            creationflags = getattr(subprocess_mod, "CREATE_NEW_PROCESS_GROUP", 0)

        # cmd 用 /c，其它 shell 用 -c。
        shell_flag = (
            "/c"
            if os.name == "nt" and os.path.basename(shell).lower() in ("cmd", "cmd.exe")
            else "-c"
        )

        try:
            process = await asyncio.create_subprocess_exec(
                shell,
                shell_flag,
                command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
        except BaseException as error:
            return (False, to_execution_error(error))

        self._active_processes.add(process)

        async def _kill_tree() -> None:
            if process.pid is None:
                return
            if os.name == "nt":
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "taskkill",
                        "/F",
                        "/T",
                        "/PID",
                        str(process.pid),
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc.wait()
                except BaseException:
                    pass
            else:
                try:
                    os.killpg(process.pid, 9)
                except BaseException:
                    try:
                        process.kill()
                    except BaseException:
                        pass

        abort_waiter: asyncio.Task | None = None
        if options.abort_signal is not None:

            async def _on_abort() -> None:
                await options.abort_signal.wait()
                await _kill_tree()

            abort_waiter = asyncio.create_task(_on_abort())

        async def _reader(stream: asyncio.StreamReader | None, sink: list[str], callback) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                sink.append(text)
                if callback is not None:
                    callback(text)

        stdout_task = asyncio.create_task(_reader(process.stdout, stdout_chunks, options.on_stdout))
        stderr_task = asyncio.create_task(_reader(process.stderr, stderr_chunks, options.on_stderr))

        try:
            if timeout is not None:
                try:
                    exit_code = await asyncio.wait_for(process.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    timed_out = True
                    await _kill_tree()
                    exit_code = None
            else:
                exit_code = await process.wait()
        finally:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            if abort_waiter is not None:
                abort_waiter.cancel()
                try:
                    await abort_waiter
                except (asyncio.CancelledError, BaseException):
                    pass
            self._active_processes.discard(process)

        if callback_error is not None:
            return (False, callback_error)
        if timed_out:
            return (False, ExecutionError("timeout", f"timeout:{options.timeout}"))
        if options.abort_signal is not None and options.abort_signal.is_set():
            return (False, ExecutionError("aborted", "aborted"))
        return (True, ShellResult("".join(stdout_chunks), "".join(stderr_chunks), exit_code or 0))

    async def cleanup(self) -> None:
        for process in list(self._active_processes):
            try:
                process.kill()
            except BaseException:
                pass
        self._active_processes.clear()


# subprocess 模块引用（避免顶层 import 名称冲突）
import subprocess as subprocess_mod
