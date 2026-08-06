"""LaTeX 数学 → 终端友好 Unicode 文本渲染器。

对齐 TS packages/tui/src/latex.ts：支持常用符号、分数/根式/上下标、
display 布局（分数与算符上下限垂直堆叠）、矩阵/对齐环境。
表达式含不支持或残缺语法时返回 None。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rich.cells import cell_len


SYMBOLS: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "varkappa": "ϰ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "ϕ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "div": "÷",
    "cdot": "·",
    "ast": "∗",
    "star": "⋆",
    "circ": "∘",
    "bullet": "•",
    "oplus": "⊕",
    "ominus": "⊖",
    "otimes": "⊗",
    "oslash": "⊘",
    "odot": "⊙",
    "bigcirc": "○",
    "dagger": "†",
    "ddagger": "‡",
    "amalg": "⨿",
    "uplus": "⊎",
    "sqcap": "⊓",
    "sqcup": "⊔",
    "triangleleft": "◁",
    "triangleright": "▷",
    "wr": "≀",
    "cap": "∩",
    "cup": "∪",
    "bigcap": "⋂",
    "bigcup": "⋃",
    "bigwedge": "⋀",
    "bigvee": "⋁",
    "bigsqcup": "⨆",
    "biguplus": "⨄",
    "bigoplus": "⨁",
    "bigotimes": "⨂",
    "bigodot": "⨀",
    "setminus": "∖",
    "in": "∈",
    "notin": "∉",
    "ni": "∋",
    "subset": "⊂",
    "supset": "⊃",
    "subseteq": "⊆",
    "supseteq": "⊇",
    "sqsubset": "⊏",
    "sqsupset": "⊐",
    "sqsubseteq": "⊑",
    "sqsupseteq": "⊒",
    "prec": "≺",
    "preceq": "≼",
    "succ": "≻",
    "succeq": "≽",
    "ll": "≪",
    "gg": "≫",
    "le": "≤",
    "leq": "≤",
    "leqslant": "≤",
    "ge": "≥",
    "geq": "≥",
    "geqslant": "≥",
    "ne": "≠",
    "neq": "≠",
    "equiv": "≡",
    "approx": "≈",
    "sim": "∼",
    "simeq": "≃",
    "cong": "≅",
    "asymp": "≍",
    "doteq": "≐",
    "propto": "∝",
    "parallel": "∥",
    "perp": "⊥",
    "mid": "∣",
    "vdash": "⊢",
    "dashv": "⊣",
    "models": "⊨",
    "Vdash": "⊩",
    "Vvdash": "⊪",
    "nvdash": "⊬",
    "nvDash": "⊭",
    "forall": "∀",
    "exists": "∃",
    "nexists": "∄",
    "neg": "¬",
    "land": "∧",
    "wedge": "∧",
    "lor": "∨",
    "vee": "∨",
    "to": "→",
    "rightarrow": "→",
    "longrightarrow": "→",
    "leftarrow": "←",
    "longleftarrow": "←",
    "gets": "←",
    "leftrightarrow": "↔",
    "longleftrightarrow": "↔",
    "hookleftarrow": "↩",
    "hookrightarrow": "↪",
    "twoheadleftarrow": "↞",
    "twoheadrightarrow": "↠",
    "leftharpoonup": "↼",
    "leftharpoondown": "↽",
    "rightharpoonup": "⇀",
    "rightharpoondown": "⇁",
    "rightleftharpoons": "⇌",
    "leftrightharpoons": "⇋",
    "nearrow": "↗",
    "searrow": "↘",
    "swarrow": "↙",
    "nwarrow": "↖",
    "rightsquigarrow": "⇝",
    "leadsto": "⇝",
    "Rightarrow": "⇒",
    "Longrightarrow": "⇒",
    "Leftarrow": "⇐",
    "Longleftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "Longleftrightarrow": "⇔",
    "implies": "⇒",
    "iff": "⇔",
    "mapsto": "↦",
    "longmapsto": "↦",
    "uparrow": "↑",
    "downarrow": "↓",
    "partial": "∂",
    "nabla": "∇",
    "int": "∫",
    "iint": "∬",
    "iiint": "∭",
    "oint": "∮",
    "sum": "∑",
    "prod": "∏",
    "coprod": "∐",
    "infty": "∞",
    "emptyset": "∅",
    "varnothing": "∅",
    "angle": "∠",
    "therefore": "∴",
    "because": "∵",
    "aleph": "ℵ",
    "beth": "ℶ",
    "gimel": "ℷ",
    "daleth": "ℸ",
    "top": "⊤",
    "bot": "⊥",
    "triangle": "△",
    "square": "□",
    "lozenge": "◊",
    "checkmark": "✓",
    "complement": "∁",
    "wp": "℘",
    "prime": "′",
    "ldots": "…",
    "dots": "…",
    "cdots": "⋯",
    "vdots": "⋮",
    "ddots": "⋱",
    "ell": "ℓ",
    "hbar": "ℏ",
    "Im": "ℑ",
    "Re": "ℜ",
    "langle": "⟨",
    "rangle": "⟩",
    "vert": "|",
    "lvert": "|",
    "rvert": "|",
    "Vert": "‖",
    "lVert": "‖",
    "rVert": "‖",
    "lbrace": "{",
    "rbrace": "}",
    "backslash": "\\",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "colon": ":",
}

NAMED_OPERATORS = {
    "arccos",
    "arcsin",
    "arctan",
    "arg",
    "cos",
    "cosh",
    "cot",
    "coth",
    "csc",
    "deg",
    "det",
    "dim",
    "exp",
    "gcd",
    "hom",
    "inf",
    "ker",
    "lg",
    "lim",
    "liminf",
    "limsup",
    "ln",
    "log",
    "max",
    "min",
    "Pr",
    "sec",
    "sin",
    "sinh",
    "sup",
    "tan",
    "tanh",
}

LIMIT_OPERATORS = {
    "argmax",
    "argmin",
    "inf",
    "injlim",
    "lim",
    "liminf",
    "limsup",
    "max",
    "min",
    "projlim",
    "sup",
}

DISPLAY_LIMIT_SYMBOLS = {
    "bigcap",
    "bigcup",
    "bigodot",
    "bigoplus",
    "bigotimes",
    "bigsqcup",
    "biguplus",
    "bigvee",
    "bigwedge",
    "coprod",
    "int",
    "iint",
    "iiint",
    "oint",
    "prod",
    "sum",
}

NEGATED_SYMBOLS: dict[str, str] = {
    "<": "≮",
    ">": "≯",
    "=": "≠",
    "∈": "∉",
    "∋": "∌",
    "∣": "∤",
    "∥": "∦",
    "∼": "≁",
    "≃": "≄",
    "≅": "≇",
    "≈": "≉",
    "≡": "≢",
    "≤": "≰",
    "≥": "≱",
    "≺": "⊀",
    "≻": "⊁",
    "⊂": "⊄",
    "⊃": "⊅",
    "⊆": "⊈",
    "⊇": "⊉",
    "⊢": "⊬",
    "⊨": "⊭",
    "↔": "↮",
    "←": "↚",
    "→": "↛",
    "⇒": "⇏",
    "⇐": "⇍",
    "⇔": "⇎",
    "≼": "⋠",
    "≽": "⋡",
}

BLACKBOARD: dict[str, str] = {
    "C": "ℂ",
    "H": "ℍ",
    "N": "ℕ",
    "P": "ℙ",
    "Q": "ℚ",
    "R": "ℝ",
    "Z": "ℤ",
}

SUPERSCRIPTS: dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}

SUBSCRIPTS: dict[str, str] = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
}

SPACING_COMMANDS = {
    ",",
    ":",
    ";",
    " ",
    ">",
    "enspace",
    "enskip",
    "medspace",
    "quad",
    "qquad",
    "thickspace",
    "thinspace",
}
NEGATIVE_SPACING_COMMANDS = {"!", "negmedspace", "negthickspace", "negthinspace"}
NEGATIVE_SPACE = "\u0000"
IGNORED_COMMANDS = {
    "displaystyle",
    "limits",
    "nolimits",
    "scriptstyle",
    "scriptscriptstyle",
    "textstyle",
}
SIZE_COMMANDS = {
    "big",
    "Big",
    "bigg",
    "Bigg",
    "bigl",
    "Bigl",
    "biggl",
    "Biggl",
    "bigr",
    "Bigr",
    "biggr",
    "Biggr",
}
PLAIN_WRAPPERS = {
    "emph",
    "mathcal",
    "mathbf",
    "mathfrak",
    "mathit",
    "mathrm",
    "mathnormal",
    "mathscr",
    "mathsf",
    "mathtt",
    "mathup",
    "mbox",
    "overbrace",
    "pmb",
    "smash",
    "substack",
    "text",
    "textbf",
    "textit",
    "textmd",
    "textnormal",
    "textrm",
    "textsc",
    "textsf",
    "textsl",
    "texttt",
    "textup",
    "underbrace",
    "bm",
    "boldsymbol",
}
ACCENTS: dict[str, str] = {
    "acute": "\u0301",
    "bar": "\u0305",
    "breve": "\u0306",
    "check": "\u030c",
    "ddot": "\u0308",
    "dot": "\u0307",
    "grave": "\u0300",
    "hat": "\u0302",
    "mathring": "\u030a",
    "overleftarrow": "\u20d6",
    "overleftrightarrow": "\u20e1",
    "overline": "\u0305",
    "overrightarrow": "\u20d7",
    "tilde": "\u0303",
    "underline": "\u0332",
    "vec": "\u20d7",
    "widehat": "\u0302",
    "widetilde": "\u0303",
}


def _visible_width(value: str) -> int:
    return cell_len(value)


def _replace_characters(value: str, replacements: dict[str, str]) -> str | None:
    result: list[str] = []
    for character in value:
        replacement = replacements.get(character)
        if replacement is None:
            return None
        result.append(replacement)
    return "".join(result)


def _format_script(value: str, kind: str) -> str:
    value = value.strip()
    replacements = SUBSCRIPTS if kind == "sub" else SUPERSCRIPTS
    unicode_result = _replace_characters(value, replacements)
    if unicode_result is not None:
        return unicode_result
    prefix = "_" if kind == "sub" else "^"
    if len(value) == 1 or (kind == "sub" and value.isascii() and value.isalpha()):
        return f"{prefix}{value}"
    return f"{prefix}({value})"


_SIMPLE_TEXT = re.compile(r"^[\w.]+$", re.UNICODE)


def _format_fraction(numerator: str, denominator: str) -> str:
    numerator = numerator.strip()
    denominator = denominator.strip()
    simple_numerator = _SIMPLE_TEXT.match(numerator) is not None
    simple_denominator = _SIMPLE_TEXT.match(denominator) is not None or len(denominator) == 1
    num = numerator if simple_numerator else f"({numerator})"
    den = denominator if simple_denominator else f"({denominator})"
    return f"{num}/{den}"


def _format_root(value: str, symbol: str = "√") -> str:
    value = value.strip()
    return f"{symbol}{value}" if _SIMPLE_TEXT.match(value) else f"{symbol}({value})"


def _normalize_output(value: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    filtered: list[str] = []
    for index, line in enumerate(lines):
        if len(line) > 0 or (0 < index < len(lines) - 1):
            filtered.append(line)
    return "\n".join(filtered).strip()


@dataclass
class FractionNode:
    type: Literal["fraction"] = "fraction"
    numerator: str = ""
    denominator: str = ""


@dataclass
class OperatorNode:
    type: Literal["operator"] = "operator"
    operator: str = ""
    lower: str | None = None
    upper: str | None = None


LayoutNode = FractionNode | OperatorNode


@dataclass
class Layout:
    lines: list[str]
    width: int
    baseline: int


LAYOUT_MARKER_START = "\uf0000"
LAYOUT_MARKER_END = "\uf0001"
LAYOUT_MARKER_PATTERN = re.compile(r"\uf0000(\d+)\uf0001")
PROTECTED_SPACE = "\u00a0"


def _pad_layout_line(line: str, width: int, centered: bool = False) -> str:
    padding = max(0, width - _visible_width(line))
    left = padding // 2 if centered else 0
    return " " * left + line + " " * (padding - left)


def _join_layouts(layouts: list[Layout]) -> Layout:
    if not layouts:
        return Layout(lines=[""], width=0, baseline=0)
    baseline = max(layout.baseline for layout in layouts)
    below = max(layout.lines.__len__() - layout.baseline - 1 for layout in layouts)
    lines: list[str] = []
    for row in range(baseline + below + 1):
        line = ""
        for layout in layouts:
            source_row = row - baseline + layout.baseline
            if 0 <= source_row < len(layout.lines):
                line += _pad_layout_line(layout.lines[source_row], layout.width)
            else:
                line += " " * layout.width
        lines.append(line.rstrip())
    return Layout(
        lines=lines,
        width=sum(layout.width for layout in layouts),
        baseline=baseline,
    )


def _render_layout(source: str, nodes: list[LayoutNode]) -> Layout:
    rendered_lines: list[str] = []
    first_baseline = 0
    for source_line in source.split("\n"):
        layouts: list[Layout] = []
        position = 0
        previous_was_node = False
        for match in LAYOUT_MARKER_PATTERN.finditer(source_line):
            index = match.start()
            if index > position:
                sliced = source_line[position:index]
                text = (sliced.lstrip() if previous_was_node else sliced).rstrip()
                layouts.append(Layout(lines=[text], width=_visible_width(text), baseline=0))
            node_index = int(match.group(1))
            node = nodes[node_index] if node_index < len(nodes) else None
            if node is not None:
                if node.type == "fraction":
                    numerator = _render_layout(node.numerator, nodes)
                    denominator = _render_layout(node.denominator, nodes)
                    content_width = max(numerator.width, denominator.width, 1)
                    width = content_width + 2
                    fraction_lines = [
                        _pad_layout_line(line, width, True) for line in numerator.lines
                    ]
                    fraction_lines.append(f" {'─' * content_width} ")
                    fraction_lines.extend(
                        _pad_layout_line(line, width, True) for line in denominator.lines
                    )
                    layouts.append(
                        Layout(
                            lines=fraction_lines,
                            width=width,
                            baseline=len(numerator.lines),
                        )
                    )
                else:
                    content_width = max(
                        _visible_width(node.operator),
                        0 if node.lower is None else _visible_width(node.lower),
                        0 if node.upper is None else _visible_width(node.upper),
                    )
                    lines: list[str] = []
                    if node.upper is not None:
                        lines.append(f"{_pad_layout_line(node.upper, content_width, True)} ")
                    lines.append(f"{_pad_layout_line(node.operator, content_width, True)} ")
                    if node.lower is not None:
                        lines.append(f"{_pad_layout_line(node.lower, content_width, True)} ")
                    layouts.append(
                        Layout(
                            lines=lines,
                            width=content_width + 1,
                            baseline=0 if node.upper is None else 1,
                        )
                    )
            position = index + len(match.group(0))
            previous_was_node = True
        if position < len(source_line):
            sliced = source_line[position:]
            text = sliced.lstrip() if previous_was_node else sliced
            layouts.append(Layout(lines=[text], width=_visible_width(text), baseline=0))
        line_layout = _join_layouts(layouts)
        if not rendered_lines:
            first_baseline = line_layout.baseline
        rendered_lines.extend(line_layout.lines)
    return Layout(
        lines=rendered_lines,
        width=max([0, *(_visible_width(line) for line in rendered_lines)]),
        baseline=first_baseline,
    )


class LatexParser:
    """逐字符解析 LaTeX 子集（与 TS LatexParser 同语义）。"""

    def __init__(self, source: str, layout_nodes: list[LayoutNode] | None = None) -> None:
        self.source = source
        self.layout_nodes = layout_nodes
        self.position = 0
        self.supported = True
        self.stack_fractions = True

    def render(self) -> str | None:
        rendered = self._parse_sequence()
        if not self.supported or self.position != len(self.source):
            return None
        return _normalize_output(rendered)

    def _parse_sequence(self, end_character: str | None = None) -> str:
        result: list[str] = []
        while self.position < len(self.source):
            character = self.source[self.position]
            if end_character is not None and character == end_character:
                self.position += 1
                return "".join(result)
            if character == "}":
                self.supported = False
                return "".join(result)
            if character == "{":
                self.position += 1
                result.append(self._parse_sequence("}"))
                continue
            if character == "\\":
                command = self._parse_command()
                if command == NEGATIVE_SPACE:
                    result = [part.rstrip() for part in result]
                else:
                    result.append(command)
                continue
            if character in ("^", "_"):
                self.position += 1
                result = [part.rstrip() for part in result]
                result.append(
                    _format_script(
                        self._parse_required_argument(False), "sub" if character == "_" else "sup"
                    )
                )
                continue
            if character.isspace():
                result.append(self._parse_whitespace())
                continue
            if character == "&":
                self.position += 1
                continue
            if character == "~":
                self.position += 1
                result.append(" ")
                continue
            result.append(character)
            self.position += 1
        if end_character is not None:
            self.supported = False
        return "".join(result)

    def _parse_whitespace(self) -> str:
        while self.position < len(self.source) and self.source[self.position].isspace():
            self.position += 1
        return " "

    def _parse_command(self) -> str:
        self.position += 1
        if self.position >= len(self.source):
            self.supported = False
            return ""
        first = self.source[self.position]
        if first.isascii() and first.isalpha():
            start = self.position
            while (
                self.position < len(self.source)
                and self.source[self.position].isascii()
                and self.source[self.position].isalpha()
            ):
                self.position += 1
            command = self.source[start : self.position]
        else:
            command = first
            self.position += 1

        if command == "\\":
            return "\n"
        if command in SPACING_COMMANDS:
            return " "
        if command in NEGATIVE_SPACING_COMMANDS:
            return NEGATIVE_SPACE
        if command in IGNORED_COMMANDS:
            return ""
        if command in ("{", "}", "$", "%", "#", "_", "&"):
            return command
        if command == "|":
            return "‖"
        if command == "not":
            value = self._parse_required_argument(False).strip()
            negated = NEGATED_SYMBOLS.get(value)
            if negated is not None:
                return negated
            if not value:
                self.supported = False
                return ""
            return value[0] + "\u0338" + value[1:]
        if command in LIMIT_OPERATORS:
            return self._parse_operator(command, "bracket", True, True)

        symbol = SYMBOLS.get(command)
        if symbol is not None:
            return (
                self._parse_operator(symbol, "script", True)
                if command in DISPLAY_LIMIT_SYMBOLS
                else symbol
            )
        if command in NAMED_OPERATORS:
            return f" {command} "
        if command in SIZE_COMMANDS:
            return ""
        if command in ("left", "middle", "right"):
            if self.position < len(self.source) and self.source[self.position] == ".":
                self.position += 1
            return ""
        if command in ("frac", "dfrac", "tfrac"):
            should_stack = (
                self.layout_nodes is not None and self.stack_fractions and command != "tfrac"
            )
            numerator = self._parse_required_argument(not should_stack)
            denominator = self._parse_required_argument(not should_stack)
            if should_stack and self.layout_nodes is not None:
                self.layout_nodes.append(
                    FractionNode(
                        numerator=_normalize_output(numerator),
                        denominator=_normalize_output(denominator),
                    )
                )
                return f"{LAYOUT_MARKER_START}{len(self.layout_nodes) - 1}{LAYOUT_MARKER_END}"
            return _format_fraction(numerator, denominator)
        if command == "sqrt":
            degree = self._parse_optional_argument()
            degree = degree.strip() if degree is not None else None
            value = self._parse_required_argument()
            if degree is None or degree == "2":
                return _format_root(value)
            if degree == "3":
                return _format_root(value, "∛")
            if degree == "4":
                return _format_root(value, "∜")
            return f"{_format_script(degree, 'sup')}{_format_root(value)}"
        if command in ("boxed", "fbox"):
            return f"[{self._parse_required_argument().strip()}]"
        if command in ("binom", "dbinom", "tbinom"):
            return f"({self._parse_required_argument()} choose {self._parse_required_argument()})"
        accent = ACCENTS.get(command)
        if accent is not None:
            value = self._parse_required_argument()
            return f"{value}{accent}" if len(value) == 1 else f"{command}({value})"
        if command == "mathbb":
            value = self._parse_required_argument()
            return "".join(BLACKBOARD.get(character, character) for character in value)
        if command == "operatorname":
            starred = self.position < len(self.source) and self.source[self.position] == "*"
            if starred:
                self.position += 1
            operator = _normalize_output(self._parse_required_argument()).strip()
            return self._parse_operator(operator, "bracket", starred, True)
        if command in ("mod", "bmod"):
            return " mod "
        if command in ("pmod", "pod"):
            value = self._parse_required_argument().strip()
            return f" (mod {value})" if command == "pmod" else f" ({value})"
        if command in ("overset", "stackrel"):
            upper = self._parse_required_argument()
            value = self._parse_required_argument().strip()
            return f"{value}{_format_script(upper, 'sup')}"
        if command == "underset":
            lower = self._parse_required_argument()
            value = self._parse_required_argument().strip()
            return f"{value}{_format_script(lower, 'sub')}"
        if command in PLAIN_WRAPPERS:
            value = self._parse_required_argument()
            return value if command.startswith("text") or command == "mbox" else value.strip()
        if command == "begin":
            return self._parse_environment()
        if command == "end":
            self.supported = False
            return ""

        self.supported = False
        return f"\\{command}"

    def _parse_operator(
        self,
        operator: str,
        inline_lower_style: str,
        display_limits: bool,
        spaced: bool = False,
    ) -> str:
        use_display_limits = display_limits
        modifier_position = self.position
        while modifier_position < len(self.source) and self.source[modifier_position] in " \t":
            modifier_position += 1
        modifier_match = re.match(
            r"\\(limits|nolimits)(?![A-Za-z])", self.source[modifier_position:]
        )
        if modifier_match:
            use_display_limits = modifier_match.group(1) == "limits"
            self.position = modifier_position + len(modifier_match.group(0))

        lower: str | None = None
        upper: str | None = None
        while True:
            script_position = self.position
            while script_position < len(self.source) and self.source[script_position] in " \t":
                script_position += 1
            if script_position >= len(self.source):
                break
            kind = self.source[script_position]
            if kind not in ("_", "^"):
                break
            self.position = script_position + 1
            value = _normalize_output(self._parse_required_argument(False)).replace(" ", "")
            if kind == "_":
                if lower is not None:
                    self.supported = False
                lower = value
            else:
                if upper is not None:
                    self.supported = False
                upper = value

        if (
            self.layout_nodes is not None
            and use_display_limits
            and (lower is not None or upper is not None)
        ):
            self.layout_nodes.append(OperatorNode(operator=operator, lower=lower, upper=upper))
            return f"{LAYOUT_MARKER_START}{len(self.layout_nodes) - 1}{LAYOUT_MARKER_END}"

        rendered = operator
        if lower is not None:
            rendered += (
                f"[{lower}]" if inline_lower_style == "bracket" else _format_script(lower, "sub")
            )
        if upper is not None:
            rendered += _format_script(upper, "sup")
        return f" {rendered} " if spaced else rendered

    def _parse_required_argument(self, stack_fractions: bool = True) -> str:
        previous_stack_fractions = self.stack_fractions
        self.stack_fractions = previous_stack_fractions and stack_fractions
        value = self._parse_required_argument_value()
        self.stack_fractions = previous_stack_fractions
        return value

    def _parse_required_argument_value(self) -> str:
        while self.position < len(self.source) and self.source[self.position] in " \t":
            self.position += 1
        if self.position >= len(self.source):
            self.supported = False
            return ""
        if self.source[self.position] == "{":
            self.position += 1
            return self._parse_sequence("}")
        if self.source[self.position] == "\\":
            return self._parse_command()
        value = self.source[self.position]
        self.position += 1
        return value

    def _parse_optional_argument(self) -> str | None:
        while self.position < len(self.source) and self.source[self.position] in " \t":
            self.position += 1
        if self.position >= len(self.source) or self.source[self.position] != "[":
            return None
        end = self.source.find("]", self.position + 1)
        if end < 0:
            self.supported = False
            return None
        value = self.source[self.position + 1 : end]
        self.position = end + 1
        return self._render_nested(value)

    def _read_raw_group(self) -> str | None:
        while self.position < len(self.source) and self.source[self.position] in " \t":
            self.position += 1
        if self.position >= len(self.source) or self.source[self.position] != "{":
            self.supported = False
            return None
        start = self.position + 1
        self.position += 1
        depth = 1
        while self.position < len(self.source):
            character = self.source[self.position]
            if character == "\\":
                self.position += 2
                continue
            if character == "{":
                depth += 1
            if character == "}":
                depth -= 1
            if depth == 0:
                value = self.source[start : self.position]
                self.position += 1
                return value
            self.position += 1
        self.supported = False
        return None

    def _split_environment_rows(self, body: str) -> list[str]:
        return re.split(r"\\\\(?:\[[^\]\n]*\])?", body)

    def _parse_environment(self) -> str:
        environment = self._read_raw_group()
        if environment is None:
            return ""
        end_marker = f"\\end{{{environment}}}"
        end = self.source.find(end_marker, self.position)
        if end < 0:
            self.supported = False
            return ""
        body = self.source[self.position : end]
        self.position = end + len(end_marker)

        if environment in ("equation", "equation*", "displaymath"):
            return self._render_nested(body).strip()

        if environment in (
            "aligned",
            "align",
            "align*",
            "alignedat",
            "alignat",
            "alignat*",
            "gather",
            "gathered",
            "multline",
            "multline*",
            "split",
        ):
            aligned_at = environment in ("alignedat", "alignat", "alignat*")
            aligned_body = re.sub(r"^\s*\{[^}]*\}", "", body) if aligned_at else body
            rows: list[str] = []
            for raw_row in self._split_environment_rows(aligned_body):
                cells = raw_row.split("&")
                source = (
                    " ".join(
                        "".join(cells[index * 2 : index * 2 + 2])
                        for index in range((len(cells) + 1) // 2)
                    )
                    if aligned_at
                    else "".join(cells)
                )
                rendered_row = self._render_nested(source).strip()
                if rendered_row:
                    rows.append(rendered_row)
            return "\n".join(rows)

        if environment in ("cases", "cases*"):
            case_rows: list[list[str]] = []
            for raw_row in self._split_environment_rows(body):
                cells = [self._render_nested(cell, False).strip() for cell in raw_row.split("&")]
                if any(cells):
                    case_rows.append(cells)
            result: list[str] = []
            for index, row in enumerate(case_rows):
                value = re.sub(r",\s*$", "", row[0] if row else "")
                condition = row[1] if len(row) > 1 else ""
                delimiter = "⎧" if index == 0 else ("⎩" if index == len(case_rows) - 1 else "⎨")
                condition_prefix = (
                    " " if re.match(r"^(?:if|when|for|otherwise)\b", condition, re.I) else " if "
                )
                result.append(
                    f"{delimiter} {value}{condition_prefix + condition if condition else ''}"
                )
            return "\n".join(result)

        if environment in (
            "array",
            "matrix",
            "smallmatrix",
            "pmatrix",
            "bmatrix",
            "Bmatrix",
            "vmatrix",
            "Vmatrix",
        ):
            matrix_body = re.sub(r"^\s*\{[^}]*\}", "", body) if environment == "array" else body
            return self._render_matrix(environment, matrix_body)

        self.supported = False
        return body

    def _render_matrix(self, environment: str, body: str) -> str:
        matrix: list[list[str]] = []
        for raw_row in self._split_environment_rows(body):
            row = [self._render_nested(cell, False).strip() for cell in raw_row.split("&")]
            if any(row):
                matrix.append(row)
        column_count = max((len(row) for row in matrix), default=0)
        column_widths = [
            max((_visible_width(row[column]) for row in matrix if column < len(row)), default=0)
            for column in range(column_count)
        ]
        rows: list[str] = []
        for row in matrix:
            cells: list[str] = []
            for column in range(column_count):
                cell = row[column] if column < len(row) else ""
                cells.append(
                    cell + PROTECTED_SPACE * max(0, column_widths[column] - _visible_width(cell))
                )
            rows.append(" │ ".join(cells))
        if environment in ("array", "matrix", "smallmatrix"):
            return "\n".join(rows)

        delimiters: dict[str, tuple[str, str, str, str, str, str]] = {
            "pmatrix": ("⎛", "⎞", "⎜", "⎟", "⎝", "⎠"),
            "bmatrix": ("⎡", "⎤", "⎢", "⎥", "⎣", "⎦"),
            "Bmatrix": ("⎧", "⎫", "⎨", "⎬", "⎩", "⎭"),
            "vmatrix": ("│", "│", "│", "│", "│", "│"),
            "Vmatrix": ("║", "║", "║", "║", "║", "║"),
        }
        delimiter = delimiters.get(environment)
        if delimiter is None:
            self.supported = False
            return "\n".join(rows)
        if len(rows) == 1:
            return f"{delimiter[0]} {rows[0]} {delimiter[1]}"
        result: list[str] = []
        for index, matrix_row in enumerate(rows):
            left = (
                delimiter[0]
                if index == 0
                else (delimiter[4] if index == len(rows) - 1 else delimiter[2])
            )
            right = (
                delimiter[1]
                if index == 0
                else (delimiter[5] if index == len(rows) - 1 else delimiter[3])
            )
            result.append(f"{left} {matrix_row} {right}")
        return "\n".join(result)

    def _render_nested(self, source: str, stack_fractions: bool = True) -> str:
        rendered = LatexParser(
            source,
            self.layout_nodes if stack_fractions else None,
        ).render()
        if rendered is None:
            self.supported = False
            return source
        return rendered


def render_latex(source: str, *, display: bool = False) -> str | None:
    """渲染 LaTeX 数学表达式为 Unicode 文本；不支持/残缺时返回 None。"""
    layout_nodes: list[LayoutNode] | None = [] if display else None
    rendered = LatexParser(source, layout_nodes).render()
    if rendered is None:
        return None
    if not layout_nodes:
        return rendered.replace(PROTECTED_SPACE, " ")
    lines = _render_layout(rendered, layout_nodes).lines
    non_empty = [line for line in lines if line.strip()]
    indentation = min((len(line) - len(line.lstrip()) for line in non_empty), default=0)
    return (
        "\n".join(line[indentation:].rstrip() for line in lines)
        .rstrip()
        .replace(PROTECTED_SPACE, " ")
    )


__all__ = ["render_latex"]
