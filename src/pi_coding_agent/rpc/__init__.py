"""RPC 模式（无头协议）——对齐 TS modes/rpc/。

通过 stdin/stdout JSONL 与外部宿主（IDE 插件等）通信。
"""

from .rpc_client import RpcClient, RpcClientOptions
from .rpc_mode import RpcMessageHandler, RpcUiContext, run_rpc_mode
from .rpc_types import RpcSessionState

__all__ = [
    "RpcClient",
    "RpcClientOptions",
    "RpcMessageHandler",
    "RpcUiContext",
    "run_rpc_mode",
    "RpcSessionState",
]
