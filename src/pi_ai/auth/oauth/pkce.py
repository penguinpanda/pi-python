"""PKCE 辅助工具（对齐 TS auth/oauth/pkce.ts）。"""

import base64
import hashlib
import secrets


def generate_pkce() -> tuple[str, str]:
    """生成 (verifier, challenge)；challenge 为 verifier 的 SHA-256 base64url。"""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


__all__ = ["generate_pkce"]
