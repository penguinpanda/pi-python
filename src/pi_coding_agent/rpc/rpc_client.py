"""RPC 客户端（对齐 TS modes/rpc/rpc-client.ts）。

启动 `pi-python --mode rpc` 子进程，通过 stdin/stdout JSONL 通信。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Callable, TypedDict

from .jsonl import read_jsonl_lines, serialize_json_line


class RpcClientOptions(TypedDict, total=False):
    """RpcClient 启动选项。"""

    # 完整的子进程命令（优先级最高；默认 `python -m pi_coding_agent --mode rpc`）。
    command: list[str]
    # CLI 入口（默认 "pi-python"；仅在未指定 command 时与 provider/model/args 组合）。
    pi_path: str
    # 子进程工作目录。
    cwd: str
    # 额外环境变量（合并到当前环境）。
    env: dict[str, str]
    # 初始 provider / model。
    provider: str
    model: str
    # 附加 CLI 参数。
    args: list[str]


RpcEventListener = Callable[[dict[str, Any]], None]

_REQUEST_TIMEOUT = 30.0


class RpcClient:
    """编程式访问 coding-agent RPC 模式的客户端。"""

    def __init__(self, options: RpcClientOptions | None = None) -> None:
        self._options: RpcClientOptions = options or {}
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._exit_task: asyncio.Task | None = None
        self._event_listeners: list[RpcEventListener] = []
        self._pending: dict[str, asyncio.Future] = {}
        self._request_id = 0
        self._stderr = ""
        self._exit_error: Exception | None = None

    def _fail_pending(self, error: Exception) -> None:
        """失败所有等待中的请求,避免调用方悬挂到超时。"""
        for pending in list(self._pending.values()):
            if not pending.done():
                pending.set_exception(error)
        self._pending.clear()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """启动 RPC 子进程。"""
        if self._process is not None:
            raise RuntimeError("Client already started")

        self._exit_error = None
        command = self._build_command()
        env = dict(os.environ)
        env.update(self._options.get("env") or {})
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._options.get("cwd"),
            env=env,
        )
        self._process = process

        async def _read_stderr() -> None:
            assert process.stderr is not None
            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                self._stderr += text
                sys.stderr.write(text)
                sys.stderr.flush()

        async def _read_stdout() -> None:
            assert process.stdout is not None
            try:
                async for line in read_jsonl_lines(process.stdout):
                    self._handle_line(line)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # 读取失败（子进程崩溃、管道断裂）时失败所有等待中的请求，
                # 而不是静默吞掉让调用方空等超时。
                self._exit_error = RuntimeError(
                    f"Agent process stdout closed: {error}. Stderr: {self._stderr}"
                )
                self._fail_pending(self._exit_error)

        async def _wait_for_exit() -> None:
            code = await process.wait()
            if self._process is not None:
                # stop() 已把 _process 置 None 时不再覆盖其正常终止语义。
                self._exit_error = RuntimeError(
                    f"Agent process exited (code={code}). Stderr: {self._stderr}"
                )
                self._fail_pending(self._exit_error)

        self._reader_task = asyncio.create_task(_read_stdout())
        self._stderr_task = asyncio.create_task(_read_stderr())
        self._exit_task = asyncio.create_task(_wait_for_exit())

    async def stop(self) -> None:
        """终止 RPC 子进程。"""
        process = self._process
        if process is None:
            return
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None
        if process.returncode is None:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=2)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
        if self._exit_task is not None:
            self._exit_task.cancel()
            try:
                await self._exit_task
            except asyncio.CancelledError:
                pass
            self._exit_task = None
        self._process = None
        # stop() 后等待中的请求立即失败,而不是悬挂。
        self._fail_pending(RuntimeError("RPC client stopped"))

    def _build_command(self) -> list[str]:
        options = self._options
        command = options.get("command")
        if command:
            return list(command)
        pi_path = options.get("pi_path", "pi-python")
        if pi_path == "pi-python":
            # 默认使用当前解释器运行本包（开发/测试环境无需安装 CLI）。
            return [
                sys.executable,
                "-m",
                "pi_coding_agent",
                "--mode",
                "rpc",
                *self._extra_args(),
            ]
        return [pi_path, "--mode", "rpc", *self._extra_args()]

    def _extra_args(self) -> list[str]:
        args: list[str] = []
        provider = self._options.get("provider")
        if provider:
            args += ["--provider", provider]
        model = self._options.get("model")
        if model:
            args += ["--model", model]
        args += list(self._options.get("args") or [])
        return args

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------

    def on_event(self, listener: RpcEventListener) -> Callable[[], None]:
        """订阅 Agent 事件；返回取消订阅函数。"""
        self._event_listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._event_listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    def get_stderr(self) -> str:
        return self._stderr

    # ------------------------------------------------------------------
    # 命令方法
    # ------------------------------------------------------------------

    async def send(
        self, method: str, params: dict | None = None, timeout: float = _REQUEST_TIMEOUT
    ) -> dict:
        """发送命令并等待响应。"""
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Client not started")
        if self._exit_error is not None:
            raise self._exit_error
        if process.returncode is not None:
            raise RuntimeError(
                f"Agent process exited (code={process.returncode}). Stderr: {self._stderr}"
            )

        self._request_id += 1
        request_id = f"req_{self._request_id}"
        command = {"id": request_id, "type": method, **(params or {})}
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        try:
            process.stdin.write(serialize_json_line(command).encode("utf-8"))
            await process.stdin.drain()
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise TimeoutError(
                f"Timeout waiting for response to {method}. Stderr: {self._stderr}"
            ) from None
        finally:
            self._pending.pop(request_id, None)
        return response

    def _get_data(self, response: dict) -> Any:
        if not response.get("success"):
            raise RuntimeError(response.get("error") or "RPC command failed")
        return response.get("data")

    async def prompt(self, message: str) -> None:
        await self.send("prompt", {"message": message})

    async def steer(self, message: str) -> None:
        await self.send("steer", {"message": message})

    async def follow_up(self, message: str) -> None:
        await self.send("follow_up", {"message": message})

    async def abort(self) -> None:
        await self.send("abort")

    async def new_session(self) -> dict:
        return self._get_data(await self.send("new_session"))

    async def get_state(self) -> dict:
        return self._get_data(await self.send("get_state"))

    async def set_model(self, provider: str, model_id: str) -> dict:
        return self._get_data(
            await self.send("set_model", {"provider": provider, "modelId": model_id})
        )

    async def cycle_model(self):
        return self._get_data(await self.send("cycle_model"))

    async def get_available_models(self) -> list[dict]:
        data = self._get_data(await self.send("get_available_models"))
        return data["models"]

    async def set_thinking_level(self, level: str) -> None:
        await self.send("set_thinking_level", {"level": level})

    async def cycle_thinking_level(self):
        return self._get_data(await self.send("cycle_thinking_level"))

    async def get_available_thinking_levels(self) -> list[str]:
        data = self._get_data(await self.send("get_available_thinking_levels"))
        return data["levels"]

    async def set_steering_mode(self, mode: str) -> None:
        await self.send("set_steering_mode", {"mode": mode})

    async def set_follow_up_mode(self, mode: str) -> None:
        await self.send("set_follow_up_mode", {"mode": mode})

    async def compact(self, custom_instructions: str | None = None) -> dict:
        return self._get_data(
            await self.send("compact", {"customInstructions": custom_instructions})
        )

    async def set_auto_compaction(self, enabled: bool) -> None:
        await self.send("set_auto_compaction", {"enabled": enabled})

    async def set_auto_retry(self, enabled: bool) -> None:
        await self.send("set_auto_retry", {"enabled": enabled})

    async def abort_retry(self) -> None:
        await self.send("abort_retry")

    async def bash(self, command: str) -> dict:
        return self._get_data(await self.send("bash", {"command": command}))

    async def abort_bash(self) -> None:
        await self.send("abort_bash")

    async def get_session_stats(self) -> dict:
        return self._get_data(await self.send("get_session_stats"))

    async def export_html(self, output_path: str | None = None) -> dict:
        return self._get_data(await self.send("export_html", {"outputPath": output_path}))

    async def switch_session(self, session_path: str) -> dict:
        return self._get_data(await self.send("switch_session", {"sessionPath": session_path}))

    async def fork(self, entry_id: str) -> dict:
        return self._get_data(await self.send("fork", {"entryId": entry_id}))

    async def clone(self) -> dict:
        return self._get_data(await self.send("clone"))

    async def get_fork_messages(self) -> list[dict]:
        data = self._get_data(await self.send("get_fork_messages"))
        return data["messages"]

    async def get_entries(self, since: str | None = None) -> dict:
        return self._get_data(await self.send("get_entries", {"since": since}))

    async def get_tree(self) -> dict:
        return self._get_data(await self.send("get_tree"))

    async def get_last_assistant_text(self) -> str | None:
        data = self._get_data(await self.send("get_last_assistant_text"))
        return data["text"]

    async def set_session_name(self, name: str) -> None:
        await self.send("set_session_name", {"name": name})

    async def get_messages(self) -> list[dict]:
        data = self._get_data(await self.send("get_messages"))
        return data["messages"]

    async def get_commands(self) -> list[dict]:
        data = self._get_data(await self.send("get_commands"))
        return data["commands"]

    # ------------------------------------------------------------------
    # 等待 / 收集
    # ------------------------------------------------------------------

    async def wait_for_idle(self, timeout: float = 60.0) -> None:
        """等待 agent_settled 事件。"""
        await self._wait_for_event("agent_settled", timeout)

    async def collect_events(self, timeout: float = 60.0) -> list[dict]:
        """收集事件直到 agent_settled。"""
        events: list[dict] = []
        done = asyncio.Event()

        def _on_event(event: dict) -> None:
            events.append(event)
            if event.get("type") == "agent_settled":
                done.set()

        unsubscribe = self.on_event(_on_event)
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        finally:
            unsubscribe()
        return events

    async def prompt_and_wait(self, message: str, timeout: float = 60.0) -> list[dict]:
        """发送 prompt 并等待完成，返回全部事件。"""
        events_task = asyncio.create_task(self.collect_events(timeout))
        await self.prompt(message)
        return await events_task

    async def _wait_for_event(self, event_type: str, timeout: float) -> None:
        done = asyncio.Event()

        def _on_event(event: dict) -> None:
            if event.get("type") == event_type:
                done.set()

        unsubscribe = self.on_event(_on_event)
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        finally:
            unsubscribe()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _handle_line(self, line: str) -> None:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return

        if data.get("type") == "response" and data.get("id"):
            future = self._pending.get(data["id"])
            if future is not None and not future.done():
                self._pending.pop(data["id"], None)
                future.set_result(data)
                return

        for listener in list(self._event_listeners):
            try:
                listener(data)
            except Exception:
                pass


__all__ = ["RpcClient", "RpcClientOptions", "RpcEventListener"]
