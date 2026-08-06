"""终端输入解析：UTF-8 / CSI / SS3 / kitty 协议 / SGR 鼠标 / OSC。

对齐 TS packages/tui/src/keys.ts 的解析面（含 kitty key release）。
"""

from __future__ import annotations

from dataclasses import dataclass

_CTRL_NAMES = {
    0x00: "ctrl+space",
    0x01: "ctrl+a",
    0x02: "ctrl+b",
    0x03: "ctrl+c",
    0x04: "ctrl+d",
    0x05: "ctrl+e",
    0x06: "ctrl+f",
    0x07: "ctrl+g",
    0x08: "backspace",
    0x09: "tab",
    0x0A: "enter",
    0x0B: "ctrl+k",
    0x0C: "ctrl+l",
    0x0D: "enter",
    0x0E: "ctrl+n",
    0x0F: "ctrl+o",
    0x10: "ctrl+p",
    0x11: "ctrl+q",
    0x12: "ctrl+r",
    0x13: "ctrl+s",
    0x14: "ctrl+t",
    0x15: "ctrl+u",
    0x16: "ctrl+v",
    0x17: "ctrl+w",
    0x18: "ctrl+x",
    0x19: "ctrl+y",
    0x1A: "ctrl+z",
    0x1B: "escape",
    0x1C: "ctrl+\\",
    0x1D: "ctrl+]",
    0x1E: "ctrl+^",
    0x1F: "ctrl+-",  # 对齐 TS keys.ts：\x1f 解析为 ctrl+-（undo）
    0x7F: "backspace",
}

_CSI_KEY_CODES = {
    1: "home",
    2: "insert",
    3: "delete",
    4: "end",
    5: "pageup",
    6: "pagedown",
    7: "home",
    8: "end",
    11: "f1",
    12: "f2",
    13: "f3",
    14: "f4",
    15: "f5",
    17: "f6",
    18: "f7",
    19: "f8",
    20: "f9",
    21: "f10",
    23: "f11",
    24: "f12",
}

_MODIFIER_NAMES = {
    1: "",
    2: "shift+",
    3: "alt+",
    4: "shift+alt+",
    5: "ctrl+",
    6: "shift+ctrl+",
    7: "alt+ctrl+",
    8: "shift+alt+ctrl+",
}


def normalize_key_name(name: str) -> str:
    """规范化修饰键顺序：ctrl+shift+alt+<key>。"""
    if "+" not in name:
        return name
    parts = name.split("+")
    key = parts[-1]
    modifiers = [part for part in parts[:-1] if part]
    order = {"ctrl": 0, "shift": 1, "alt": 2, "meta": 3}
    modifiers.sort(key=lambda part: order.get(part, 4))
    return "+".join([*modifiers, key])


@dataclass(frozen=True)
class Key:
    """单个按键（name 为规范名，如 enter / ctrl+p / shift+tab / f1）。

    release=True 表示 kitty key release 事件（对齐 TS isKeyRelease）。
    """

    name: str
    char: str | None = None
    release: bool = False

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class MouseEvent:
    """SGR 鼠标事件。"""

    type: str  # press / release / motion / wheel
    button: str  # left / middle / right / up / down / none
    row: int
    col: int
    shift: bool = False
    alt: bool = False
    ctrl: bool = False


@dataclass(frozen=True)
class KeyEvent:
    """输入事件：key / paste / mouse / osc / resize。"""

    type: str
    key: Key | None = None
    text: str = ""
    mouse: MouseEvent | None = None
    data: str = ""
    width: int = 0
    height: int = 0


def _with_modifiers(base: str, modifier: int) -> str:
    """kitty modifier 位 → 名称前缀（1=shift 2=alt 4=ctrl 8=meta）。"""
    parts: list[str] = []
    if modifier & 4:
        parts.append("ctrl")
    if modifier & 1:
        parts.append("shift")
    if modifier & 2:
        parts.append("alt")
    return normalize_key_name("+".join([*parts, base])) if parts else base


class KeyParser:
    """字节流 → KeyEvent 增量解析器。"""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.paste_buffer: list[str] = []
        self.in_paste = False

    def feed(self, data: bytes, *, final: bool = False) -> list[KeyEvent]:
        """喂入字节，返回已完整的事件（不完整序列保留到下一批）。"""
        self.buffer.extend(data)
        events: list[KeyEvent] = []
        while True:
            if self.in_paste:
                end = self.buffer.find(b"\x1b[201~")
                if end < 0:
                    self.paste_buffer.append(self.buffer.decode("utf-8", "replace"))
                    self.buffer.clear()
                    break
                if end:
                    self.paste_buffer.append(bytes(self.buffer[:end]).decode("utf-8", "replace"))
                del self.buffer[: end + len(b"\x1b[201~")]
                self.in_paste = False
                events.append(KeyEvent(type="paste", text="".join(self.paste_buffer)))
                self.paste_buffer = []
                continue

            if not self.buffer:
                break
            if self.buffer[0] != 0x1B:
                event, consumed = self._parse_plain()
                if event is None:
                    if final:
                        event = KeyEvent(
                            type="key",
                            key=Key(self.buffer[0:1].decode("latin-1")),
                        )
                        consumed = 1
                    else:
                        break
                del self.buffer[:consumed]
                events.append(event)
                continue

            # ESC 开头：CSI / SS3 / OSC / alt+key / 独立 escape。
            if len(self.buffer) == 1:
                if final:
                    events.append(KeyEvent(type="key", key=Key("escape")))
                    self.buffer.clear()
                break
            second = self.buffer[1]
            if second == ord("["):
                event, consumed = self._parse_csi()
            elif second == ord("O"):
                event, consumed = self._parse_ss3()
            elif second == ord("]"):
                event, consumed = self._parse_osc()
            elif second == 0x1B:
                # 连续 ESC：先出 escape，再处理剩余。
                events.append(KeyEvent(type="key", key=Key("escape")))
                del self.buffer[:1]
                continue
            else:
                char = chr(second)
                event = KeyEvent(
                    type="key",
                    key=Key(
                        name=normalize_key_name(f"alt+{char}"),
                        char=char,
                    ),
                )
                consumed = 2
            if event is None:
                if final:
                    # 疑似未完成的转义前缀（ESC[/ESC]/ESCO）保留等待下一块；
                    # 其余无法解析的普通字节丢弃，避免卡死。
                    if (
                        self.buffer[0] != 0x1B
                        or len(self.buffer) < 2
                        or self.buffer[1]
                        not in (
                            ord("["),
                            ord("]"),
                            ord("O"),
                        )
                    ):
                        self.buffer.clear()
                break
            del self.buffer[:consumed]
            events.append(event)
        return events

    # ------------------------------------------------------------------
    # 解析器
    # ------------------------------------------------------------------

    def _parse_plain(self) -> tuple[KeyEvent | None, int]:
        """普通字节 / UTF-8 / 控制键。"""
        byte = self.buffer[0]
        if byte < 0x80:
            if byte in _CTRL_NAMES:
                return KeyEvent(type="key", key=Key(_CTRL_NAMES[byte])), 1
            return KeyEvent(type="key", key=Key(chr(byte), char=chr(byte))), 1
        # 多字节 UTF-8：尝试完整解码。
        for length in range(2, min(5, len(self.buffer) + 1)):
            candidate = bytes(self.buffer[:length])
            try:
                char = candidate.decode("utf-8")
            except UnicodeDecodeError as exc:
                if exc.reason == "unexpected end of data":
                    continue  # 需要更多字节
                return KeyEvent(type="key", key=Key(chr(byte), char=chr(byte))), 1
            return KeyEvent(type="key", key=Key(char, char=char)), length
        return None, 0

    def _parse_csi(self) -> tuple[KeyEvent | None, int]:
        """CSI 序列：ESC [ params final。"""
        index = 2
        while index < len(self.buffer):
            byte = self.buffer[index]
            if 0x40 <= byte <= 0x7E:
                break
            index += 1
        else:
            return None, 0  # 未完成
        payload = bytes(self.buffer[2:index]).decode("latin-1")
        final = chr(self.buffer[index])
        consumed = index + 1
        params = payload.split(";")

        if final in ("A", "B", "C", "D", "H", "F", "Z"):
            base = {
                "A": "up",
                "B": "down",
                "C": "right",
                "D": "left",
                "H": "home",
                "F": "end",
                "Z": "shift+tab",
            }[final]
            if final == "Z":
                return KeyEvent(type="key", key=Key("shift+tab")), consumed
            modifier = 1
            try:
                if params and params[0] in ("", "1"):
                    modifier = int(params[1]) if len(params) > 1 else 1
            except ValueError:
                pass
            prefix = _MODIFIER_NAMES.get(modifier, "")
            return KeyEvent(type="key", key=Key(f"{prefix}{base}")), consumed

        if final in ("I", "O"):
            # 焦点事件（?1004h）：CSI I = 聚焦，CSI O = 失焦。
            return (
                KeyEvent(type="focus", data="in" if final == "I" else "out"),
                consumed,
            )

        if final == "~":
            code = _first_int(payload)
            if code == 200:
                # bracketed paste 开始：进入粘贴累积模式。
                self.in_paste = True
                return KeyEvent(type="ignore"), consumed
            if code == 201:
                return KeyEvent(type="ignore"), consumed
            if code in _CSI_KEY_CODES:
                return KeyEvent(type="key", key=Key(_CSI_KEY_CODES[code])), consumed
            return KeyEvent(type="key", key=Key("unknown")), consumed

        if final == "u":
            # kitty keyboard protocol：codepoint;modifiers[:event_type] u，
            # 或协商响应 CSI ? <flags> u。
            if params[0].startswith("?"):
                try:
                    flags = int(params[0][1:])
                except ValueError:
                    return KeyEvent(type="ignore"), consumed
                return KeyEvent(type="kitty_flags", data=str(flags)), consumed
            try:
                codepoint = int(params[0])
                modifier_part = params[1] if len(params) > 1 else "1"
                modifier = int(modifier_part.split(":", 1)[0]) or 1
            except ValueError:
                return None, consumed
            release = False
            if ":" in modifier_part:
                try:
                    event_type = int(modifier_part.split(":", 1)[1].split(";")[0])
                except ValueError:
                    event_type = 1
                if event_type == 3:
                    release = True
            if codepoint == 27:
                name = _with_modifiers("escape", modifier)
            elif codepoint == 9:
                name = _with_modifiers("tab", modifier)
            elif codepoint == 13:
                name = _with_modifiers("enter", modifier)
            elif codepoint == 127:
                name = _with_modifiers("backspace", modifier)
            elif codepoint < 32:
                name = _with_modifiers(chr(ord("a") + codepoint - 1), modifier | 4)
            else:
                char = chr(codepoint)
                name = _with_modifiers(char, modifier)
            return KeyEvent(
                type="key",
                key=Key(
                    name,
                    char=chr(codepoint) if codepoint >= 32 else None,
                    release=release,
                ),
            ), consumed

        if final == "n" and params and params[0] == "?997" and len(params) >= 2:
            # 颜色方案通知：CSI ? 997 ; 1|2 n（1=dark 2=light）。
            return KeyEvent(type="color_scheme", data=params[1]), consumed

        if final in ("M", "m"):
            mouse = _parse_sgr_mouse(payload, is_release=(final == "m"))
            if mouse is not None:
                return KeyEvent(type="mouse", mouse=mouse), consumed
            return None, consumed

        # 其他 CSI（DSR 响应、DECRQM 等）：直接消费并忽略。
        return KeyEvent(type="ignore"), consumed

    def _parse_ss3(self) -> tuple[KeyEvent | None, int]:
        if len(self.buffer) < 3:
            return None, 0
        code = chr(self.buffer[2])
        name = {
            "P": "f1",
            "Q": "f2",
            "R": "f3",
            "S": "f4",
            "A": "up",
            "B": "down",
            "C": "right",
            "D": "left",
            "H": "home",
            "F": "end",
        }.get(code)
        if name is None:
            return None, 3
        return KeyEvent(type="key", key=Key(name)), 3

    def _parse_osc(self) -> tuple[KeyEvent | None, int]:
        """OSC：ESC ] ... (BEL | ESC \\ ST)。"""
        index = 2
        while index < len(self.buffer):
            byte = self.buffer[index]
            if byte == 0x07:
                data = bytes(self.buffer[2:index]).decode("latin-1")
                return KeyEvent(type="osc", data=data), index + 1
            if byte == 0x1B and index + 1 < len(self.buffer) and self.buffer[index + 1] == 0x5C:
                data = bytes(self.buffer[2:index]).decode("latin-1")
                return KeyEvent(type="osc", data=data), index + 2
            index += 1
        return None, 0


def _first_int(payload: str) -> int | None:
    try:
        return int(payload.split(";")[0])
    except (ValueError, IndexError):
        return None


def _parse_sgr_mouse(payload: str, is_release: bool = False) -> MouseEvent | None:
    """SGR 鼠标：< b;c;m (M=按下/移动，m=松开)。"""
    if not payload.startswith("<"):
        return None
    parts = payload[1:].split(";")
    if len(parts) < 3:
        return None
    try:
        code = int(parts[0])
        col = int(parts[1]) - 1  # 终端坐标 1-based → 引擎 0-based
        row = int(parts[2]) - 1
    except ValueError:
        return None
    button_bits = code & 0x03
    shift = bool(code & 0x04)
    alt = bool(code & 0x08)
    ctrl = bool(code & 0x10)
    motion = bool(code & 0x20)
    wheel = bool(code & 0x40)
    if wheel:
        event_type = "wheel"
        button = "up" if button_bits == 0 else "down"
    elif motion:
        event_type = "motion"
        button = "none"
    elif button_bits == 3 or is_release:
        event_type = "release"
        button = "none"
    else:
        event_type = "press"
        button = {0: "left", 1: "middle", 2: "right"}.get(button_bits, "none")
    return MouseEvent(
        type=event_type,
        button=button,
        row=row,
        col=col,
        shift=shift,
        alt=alt,
        ctrl=ctrl,
    )


def parse_input(data: bytes) -> list[KeyEvent]:
    """一次性解析完整字节串（测试/简单场景）。"""
    return KeyParser().feed(data, final=True)


__all__ = [
    "Key",
    "KeyEvent",
    "MouseEvent",
    "KeyParser",
    "parse_input",
    "normalize_key_name",
]
