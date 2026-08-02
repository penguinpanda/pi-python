"""时间排序 UUID v7（对齐 TS utils/uuid.ts）。"""

import os
import threading
import time
import uuid as _uuid

# 模块级状态：同一毫秒内 sequence 递增；时间回拨时沿用最后一次时间戳。
_last_ms = -1
_sequence = 0
_lock = threading.Lock()


def uuidv7() -> str:
    """生成时间排序的 UUID v7 字符串（8-4-4-4-12）。"""
    global _last_ms, _sequence

    with _lock:
        now_ms = time.time_ns() // 1_000_000
        if now_ms > _last_ms:
            _last_ms = now_ms
            _sequence = int.from_bytes(os.urandom(4), "big")
        else:
            _sequence = (_sequence + 1) & 0xFFFFFFFF
            if _sequence == 0:
                # 32 位序列耗尽：时间戳前进 1ms，避免回绕破坏排序。
                _last_ms += 1
        ts = _last_ms
        seq = _sequence
        rand = os.urandom(8)

    b = bytearray(16)
    b[0] = (ts >> 40) & 0xFF
    b[1] = (ts >> 32) & 0xFF
    b[2] = (ts >> 24) & 0xFF
    b[3] = (ts >> 16) & 0xFF
    b[4] = (ts >> 8) & 0xFF
    b[5] = ts & 0xFF
    b[6] = 0x70 | ((seq >> 28) & 0x0F)
    b[7] = (seq >> 20) & 0xFF
    b[8] = 0x80 | ((seq >> 14) & 0x3F)
    b[9] = (seq >> 6) & 0xFF
    b[10] = ((seq & 0x3F) << 2) | (rand[0] & 0x03)
    b[11] = rand[1]
    b[12] = rand[2]
    b[13] = rand[3]
    b[14] = rand[4]
    b[15] = rand[5]

    return str(_uuid.UUID(bytes=bytes(b)))


__all__ = ["uuidv7"]
