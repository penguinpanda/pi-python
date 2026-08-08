"""pi_ai.auth — 认证包入口（原 auth.py 迁移而来，保持导入兼容）。

认证(Authentication)

=========================================================
模块职责
=========================================================

本模块负责管理和解析模型 API 所需的认证信息(API Key).

整个认证流程分为三层：

    Credential(凭证)
            │
            ▼
    CredentialStore(凭证存储)
            │
            ▼
    EnvApiKeyAuth(认证策略)
            │
            ▼
    ResolvedAuth(解析结果)
            │
            ▼
      Provider 使用 API Key 发起请求


=========================================================
认证优先级
=========================================================

API Key 的解析顺序为：

    ① 已保存的凭证(CredentialStore)

            ↓

    ② 环境变量(Environment Variable)

如果两者都不存在,则抛出异常.

例如：

    store:
        openai -> sk-xxxx

    env:
        OPENAI_API_KEY=sk-yyyy

最终优先使用：

    sk-xxxx

因为用户主动保存的凭证优先级最高.


=========================================================
为什么要分层？
=========================================================

不要让 Provider 自己去读取环境变量.

例如不要写：

    api_key = os.getenv("OPENAI_API_KEY")

因为这样：

- Provider 与环境变量耦合
- 无法支持 Credential Store
- 无法统一认证逻辑

因此：

Provider 只负责：

    api_key = await resolve_api_key(...)

而认证模块负责：

- 从哪里读取
- 优先级如何
- 如何报错

这样 Provider 完全不用关心认证来源.
"""

import os
from dataclasses import dataclass
from typing import Any, Protocol

from .types import Credential

# 凭证类型


@dataclass(slots=True)
class ApiKeyCredential:
    """
    API Key 凭证.

    表示一个已经保存的认证信息.

    例如：

        ApiKeyCredential(
            key="sk-xxxx"
        )

    凭证体系已支持 OAuth（见 auth/oauth/ 与 auth/types.OAuthCredential）；
    后续新增 Azure AD / IAM Token 等类型时，扩展 Credential 联合即可，
    无需修改其它代码.
    """

    type: str = "api_key"
    key: str | None = None


class CredentialStore(Protocol):
    """
    凭证存储接口.

    EventStream 使用 Protocol,
    这里也是一样.

    任何实现下面三个接口的对象,
    都可以作为 CredentialStore.

    例如：

        InMemoryCredentialStore

        FileCredentialStore

        SQLiteCredentialStore

        KeyringCredentialStore
    """

    async def read(self, provider_id: str) -> Credential | None: ...
    async def write(self, provider_id: str, credential: ApiKeyCredential) -> None: ...
    async def delete(self, provider_id: str) -> None: ...


# 凭证解析验证


@dataclass(slots=True)
class ResolvedAuth:
    """
    认证解析结果.

    表示已经成功获得可用于请求模型的认证信息.

    source 用于说明：

        API Key 来自哪里.

    例如：

        "stored credential"

        "OPENAI_API_KEY"
    """

    api_key: str
    source: str  # "stored credential" or env var name


def env_api_key_auth(display_name: str, env_vars: list[str]) -> "EnvApiKeyAuth":
    """
    创建基于环境变量的 API Key 认证策略.

    该函数不会立即读取环境变量.

    而是返回一个 EnvApiKeyAuth,
    后续调用 resolve() 时才真正解析 API Key.

    解析优先级：

        已保存凭证

            ↓

        环境变量（按 env_vars 顺序依次检查）
    """

    return EnvApiKeyAuth(display_name=display_name, env_vars=env_vars)


@dataclass(slots=True)
class EnvApiKeyAuth:
    """
    基于环境变量的 API Key 认证策略.

    它并不保存 API Key.

    而是定义：

    ① 可以读取哪些环境变量

    ② API Key 的解析顺序

    例如：

        env_vars=[
            "OPENAI_API_KEY",
            "OPENAI_KEY"
        ]

    resolve() 时会按顺序查找.
    """

    display_name: str
    env_vars: list[str]

    def resolve(
        self,
        credential: Any = None,
    ) -> ResolvedAuth | None:
        """
        解析 API Key.

        解析顺序：

            ① CredentialStore

                    ↓

            ② 环境变量

        成功后返回：

            ResolvedAuth

        如果没有找到：

            返回 None

        注意：

        resolve() 不会抛出异常.

        是否认为缺少 API Key 是错误,
        由调用者决定.
        """
        key = (
            credential.get("key")
            if isinstance(credential, dict)
            else getattr(credential, "key", None)
        )
        if key:
            return ResolvedAuth(api_key=key, source="stored credential")

        for var in self.env_vars:
            value = os.environ.get(var)
            if value:
                return ResolvedAuth(api_key=value, source=var)

        return None


async def resolve_api_key(
    auth: EnvApiKeyAuth,
    store: CredentialStore,
    provider_id: str,
) -> str:
    """
    解析 Provider 使用的 API Key.

    工作流程：

        CredentialStore

                │

                ▼

        EnvApiKeyAuth.resolve()

                │

                ▼

        ResolvedAuth

                │

                ▼

        返回 api_key

    如果最终没有获得 API Key,

    则抛出 ValueError,
    并提示用户应该配置哪些环境变量.
    """

    credential = await store.read(provider_id)
    result = auth.resolve(credential)

    if result is None:
        vars_str = " or ".join(auth.env_vars)
        raise ValueError(
            f"No API key configured for {auth.display_name}Set {vars_str} environment variable"
        )

    return result.api_key


# ---------------------------------------------------------------------------
# 新式凭证存储（含 modify/list；沿用旧名称保持导入兼容）。
# ---------------------------------------------------------------------------

from .credential_store import (  # noqa: E402,F401
    FileCredentialStore,
    InMemoryCredentialStore,
)
