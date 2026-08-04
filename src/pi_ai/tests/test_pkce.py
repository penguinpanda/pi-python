"""PKCE 测试。"""

import base64
import hashlib

from pi_ai.auth.oauth.pkce import generate_pkce


def test_generate_pkce_verifier_format():
    verifier, challenge = generate_pkce()
    # verifier: 43 字符 base64url（32 随机字节）
    assert len(verifier) == 43
    assert set(verifier) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    # challenge: verifier 的 SHA-256 base64url（无 padding）
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    assert challenge == expected


def test_generate_pkce_unique():
    values = {generate_pkce()[0] for _ in range(50)}
    assert len(values) == 50
