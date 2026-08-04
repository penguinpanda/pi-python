"""UUID v7 测试。"""

import re

from pi_ai.utils.uuid import uuidv7

_UUID7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_uuidv7_format_and_version_bits():
    value = uuidv7()
    assert _UUID7_RE.match(value), value
    # 版本 7、变体 8/9/a/b 由正则保证。


def test_uuidv7_unique():
    values = {uuidv7() for _ in range(200)}
    assert len(values) == 200


def test_uuidv7_monotonic_within_same_ms():
    # 时间戳级排序：同一毫秒内多次调用必须严格递增。
    values = [uuidv7() for _ in range(500)]
    assert values == sorted(values)


def test_uuidv7_embeds_timestamp():
    import time

    before = int(time.time() * 1000)
    value = uuidv7()
    after = int(time.time() * 1000)
    # 前 48 位（前两段共 12 个十六进制字符）即毫秒时间戳。
    ts = int(value[:8] + value[9:13], 16)
    assert before - 1 <= ts <= after + 1
