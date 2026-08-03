"""认证类型（对齐 TS auth/types.ts）。

- Credential：api_key / oauth 两种凭证；
- CredentialStore：read/list/modify/delete；modify 是唯一写路径，
  按 provider 串行化（登录 / OAuth 刷新 / logout 都走它）；
- AuthInteraction / AuthPrompt / AuthEvent：交互式登录回调。
"""

from typing import Any, Literal, NotRequired, Protocol, TypedDict

from .context import AuthContext


class OAuthCredential(TypedDict, total=False):
    """OAuth 凭证（access/refresh/expires + provider 扩展字段）。"""

    type: Literal["oauth"]
    access: str
    refresh: str
    expires: int  # Unix 毫秒
    # 扩展字段（openai-codex account_id、github-copilot enterprise_url /
    # available_model_ids 等）直接以额外键存放。


Credential = OAuthCredential | Any  # 实际为 OAuthCredential | ApiKeyCredential


def credential_type(credential: Any) -> str | None:
    """读取凭证类型（兼容 dataclass ApiKeyCredential 与 OAuth dict）。"""
    if isinstance(credential, dict):
        return credential.get("type")
    return getattr(credential, "type", None)


class CredentialInfo(TypedDict):
    provider_id: str
    type: str


class CredentialStore(Protocol):
    """按 provider ID 的凭证存储（modify 为唯一写路径）。"""

    async def read(self, provider_id: str) -> Credential | None: ...

    async def list(self) -> list[CredentialInfo]: ...

    async def modify(
        self,
        provider_id: str,
        fn: Any,  # Callable[[Credential | None], Awaitable[Credential | None]]
    ) -> Credential | None: ...

    async def delete(self, provider_id: str) -> None: ...


# ---------------- 交互式登录 ----------------


class AuthPrompt(TypedDict, total=False):
    type: Literal["text", "secret", "select", "manual_code"]
    message: str
    placeholder: str
    options: list[dict[str, str]]


class AuthEvent(TypedDict, total=False):
    type: Literal["info", "auth_url", "device_code", "progress"]
    message: str
    url: str
    instructions: str
    user_code: str
    verification_uri: str
    interval_seconds: int
    expires_in_seconds: int


class AuthInteraction(Protocol):
    """登录流程的交互回调。"""

    signal: Any  # asyncio.Event | None

    async def prompt(self, prompt: AuthPrompt) -> str: ...

    def notify(self, event: AuthEvent) -> None: ...


class ModelAuth(TypedDict, total=False):
    """一次请求的认证结果。"""

    api_key: str
    headers: dict[str, str | None]
    base_url: str


class AuthResult:
    """认证解析结果（auth + env + source）。"""

    def __init__(
        self,
        auth: ModelAuth,
        *,
        env: dict[str, str] | None = None,
        source: str | None = None,
    ) -> None:
        self.auth = auth
        self.env = env
        self.source = source


class OAuthAuth(Protocol):
    """OAuth 认证流程（login / refresh / to_auth）。"""

    name: str

    async def login(self, interaction: AuthInteraction) -> OAuthCredential: ...

    async def refresh(
        self, credential: OAuthCredential, signal: Any = None
    ) -> OAuthCredential: ...

    async def to_auth(self, credential: OAuthCredential) -> ModelAuth: ...


__all__ = [
    "OAuthCredential",
    "Credential",
    "CredentialInfo",
    "CredentialStore",
    "credential_type",
    "AuthPrompt",
    "AuthEvent",
    "AuthInteraction",
    "ModelAuth",
    "AuthResult",
    "OAuthAuth",
    "AuthContext",
]
