"""Mermaid 代码块 → Unicode 终端图渲染（对齐 TS grok-mermaid 集成）。

内置轻量渲染器覆盖最常用的 flowchart（TD/LR）与 sequenceDiagram；
不支持的图类型返回 None（原样保留代码块 + 警告）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 图元样式类（对齐 TS grok-mermaid Span.cls）。
SPAN_BORDER = "border"
SPAN_TEXT = "text"
SPAN_EDGE = "edge"
SPAN_EDGE_LABEL = "edgeLabel"
SPAN_TITLE = "title"
SPAN_NONE = "none"


@dataclass
class _Span:
    cls: str
    text: str


@dataclass
class MermaidArt:
    """渲染结果（对齐 TS MermaidArt）。"""

    plain: list[str] = field(default_factory=list)
    styled: list[list[_Span]] = field(default_factory=list)
    width: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Node:
    id: str
    label: str
    shape: str  # box | diamond | round


@dataclass
class _Edge:
    source: str
    target: str
    label: str = ""


_EDGE_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(-->|---|-.->|-\.-|==>|===)\s*(.*?)\s*$")
_NODE_BOX_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*(\[[^\]]*\]|\{[^}]*\}|\([^)]*\))\s*$")
_NODE_BARE_RE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*$")


def _parse_flowchart(lines: list[str]) -> tuple[list[_Node], list[_Edge], list[str]]:
    nodes: dict[str, _Node] = {}
    edges: list[_Edge] = []
    warnings: list[str] = []

    def _register_node(raw: str) -> str:
        """注册节点（含形状），返回节点 id。无形状时不覆盖既有 label。"""
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(\[[^\]]*\]|\{[^}]*\}|\([^)]*\))?$", raw)
        if match:
            node_id, body = match.groups()
            shape = "box"
            label = node_id
            if body:
                if body.startswith("{"):
                    shape = "diamond"
                elif body.startswith("("):
                    shape = "round"
                label = body[1:-1].strip().strip('"')
                nodes[node_id] = _Node(id=node_id, label=label, shape=shape)
            elif node_id not in nodes:
                nodes[node_id] = _Node(id=node_id, label=node_id, shape=shape)
            return node_id
        return raw.strip()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(
            (
                "%%",
                "graph",
                "flowchart",
                "subgraph",
                "end",
                "direction",
                "classDef",
                "class",
                "style",
                "click",
                "linkStyle",
            )
        ):
            continue
        arrow = re.search(r"(-->|-\.->|==>|===|---)", stripped)
        if arrow:
            left = stripped[: arrow.start()].strip()
            right = stripped[arrow.end() :].strip()
            source = _register_node(left)
            label = ""
            # -->|label| B
            pipe = re.match(r"^\|(.+?)\|\s*(.+)$", right)
            middle = re.match(r"^(.+?)\s*-->\s*(.+)$", right)
            if pipe:
                label, target_raw = pipe.groups()
            elif middle:
                label, target_raw = middle.groups()
            else:
                target_raw = right
            target = _register_node(target_raw)
            if not source or not target:
                warnings.append(f"Could not parse edge: {stripped}")
                continue
            edges.append(_Edge(source=source, target=target, label=label.strip()))
            continue
        # 纯节点行
        box_match = _NODE_BOX_RE.match(stripped)
        if box_match:
            node_id, body = box_match.groups()
            shape = "box"
            if body.startswith("{"):
                shape = "diamond"
            elif body.startswith("("):
                shape = "round"
            nodes[node_id] = _Node(id=node_id, label=body[1:-1].strip().strip('"'), shape=shape)
            continue
        if _NODE_BARE_RE.match(stripped):
            nodes[stripped] = _Node(id=stripped, label=stripped, shape="box")
            continue
        warnings.append(f"Could not parse line: {stripped}")
    return list(nodes.values()), edges, warnings


def _frame(label: str, shape: str) -> list[str]:
    """单节点 Unicode 框（3 行；CJK 按显示宽度计算边框）。"""
    label_width = _display_width(label)
    if shape == "diamond":
        top = f"◄─{label}─►"
        return [top]
    inner = f"│ {label} │"
    top = f"┌{'─' * (label_width + 2)}┐"
    bottom = f"└{'─' * (label_width + 2)}┘"
    if shape == "round":
        top = f"╭{'─' * (label_width + 2)}╮"
        bottom = f"╰{'─' * (label_width + 2)}╯"
    return [top, inner, bottom]


def _merge_rows(rows: list[list[str]], gap: int = 2) -> list[str]:
    """把多个多行块并排拼接。"""
    if not rows:
        return []
    height = max(len(row) for row in rows)
    width = [max(_display_width(line) for line in row) for row in rows]
    out: list[str] = []
    for line_index in range(height):
        parts: list[str] = []
        for block, block_width in zip(rows, width, strict=True):
            if line_index < len(block):
                text = block[line_index]
            else:
                text = ""
            parts.append(text + " " * (block_width - _display_width(text)))
        out.append((" " * gap).join(parts).rstrip())
    return out


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if ord(char) > 0x2E80 else 1
    return width


def _flowchart_td(nodes: dict[str, _Node], edges: list[_Edge]) -> MermaidArt:
    # BFS 分层（从无入边节点开始）。
    indegree: dict[str, int] = {node.id: 0 for node in nodes.values()}
    adjacency: dict[str, list[_Edge]] = {node.id: [] for node in nodes.values()}
    for edge in edges:
        if edge.source not in adjacency:
            continue
        adjacency[edge.source].append(edge)
        if edge.target in indegree:
            indegree[edge.target] += 1

    layers: list[list[str]] = []
    remaining = dict(indegree)
    current = sorted([node_id for node_id, deg in remaining.items() if deg == 0])
    if not current and remaining:
        current = [next(iter(remaining))]
        remaining[current[0]] = -1
    while current:
        layers.append(current)
        next_layer: list[str] = []
        for node_id in current:
            for edge in adjacency.get(node_id, []):
                target = edge.target
                if target in remaining and remaining[target] > 0:
                    remaining[target] -= 1
                    if remaining[target] == 0:
                        next_layer.append(target)
        current = sorted(next_layer)

    lines: list[str] = []
    styled: list[list[_Span]] = []
    for index, layer in enumerate(layers):
        blocks = [_frame(nodes[node_id].label, nodes[node_id].shape) for node_id in layer]
        rows = _merge_rows(blocks)
        for row in rows:
            lines.append(row)
            styled.append([_Span(SPAN_BORDER, row)])
        if index < len(layers) - 1:
            down = "│" if len(layers[index + 1]) > 1 else "↓"
            lines.append(down)
            styled.append([_Span(SPAN_EDGE, down)])
    art = MermaidArt(plain=lines, styled=styled)
    art.width = max((_display_width(line) for line in lines), default=0)
    return art


def _flowchart_lr(nodes: dict[str, _Node], edges: list[_Edge]) -> MermaidArt:
    lines: list[str] = []
    styled: list[list[_Span]] = []
    node_frames = {node.id: _frame(node.label, node.shape) for node in nodes.values()}
    node_height = max((len(frame) for frame in node_frames.values()), default=1)
    # 每行一个节点 + 出边箭头
    for node in nodes.values():
        base_lines: list[str] = []
        for edge in [e for e in edges if e.source == node.id]:
            if edge.label:
                base_lines.append(f"── {edge.label} ──▶ {edge.target}")
            else:
                base_lines.append(f"──▶ {edge.target}")
        if not base_lines:
            base_lines = ["──▶"]
        frame = node_frames[node.id]
        width = max(_display_width(line) for line in frame)
        for line_index in range(max(node_height, len(base_lines))):
            left = frame[line_index] if line_index < len(frame) else " " * width
            right = base_lines[line_index] if line_index < len(base_lines) else ""
            joined = left + " " + right
            lines.append(joined.rstrip())
            styled.append([_Span(SPAN_BORDER, left), _Span(SPAN_EDGE, " " + right)])
    art = MermaidArt(plain=lines, styled=styled)
    art.width = max((_display_width(line) for line in lines), default=0)
    return art


def _sequence(lines: list[str]) -> MermaidArt:
    participants: list[tuple[str, str]] = []
    messages: list[tuple[str, str, str]] = []
    warnings: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        participant = re.match(r"^participant\s+([A-Za-z0-9_.\-]+)(?:\s+as\s+(.+))?$", stripped)
        if participant:
            alias, name = participant.groups()
            participants.append((alias, (name or alias).strip()))
            continue
        message = re.match(
            r"^([A-Za-z0-9_.\-]+)\s*(-+>>?|-->>?|->>|-->)\s*([A-Za-z0-9_.\-]+)\s*:?\s*(.*)$",
            stripped,
        )
        if message:
            source, _arrow, target, text = message.groups()
            messages.append((source, target, text.strip()))
            continue
        if stripped.startswith(
            (
                "title",
                "note",
                "activate",
                "deactivate",
                "loop",
                "alt",
                "else",
                "end",
                "opt",
                "par",
                "autonumber",
            )
        ):
            continue
        warnings.append(f"Could not parse line: {stripped}")
    if not participants:
        return MermaidArt(plain=[], warnings=warnings)
    width = max(len(name) for _alias, name in participants)
    header_parts = [f"┌{'─' * width}┐" for _ in participants]
    name_parts = [f"│ {name:<{width}} │" for _alias, name in participants]
    lines = ["  ".join(header_parts).rstrip(), "  ".join(name_parts).rstrip()]
    styled: list[list[_Span]] = [
        [_Span(SPAN_BORDER, lines[0])],
        [_Span(SPAN_TITLE, lines[1])],
    ]
    for source, target, text in messages:
        label = f" {text} " if text else ""
        row = f"{source} ──{label}──▶ {target}"
        lines.append(row)
        styled.append(
            [
                _Span(SPAN_TEXT, source),
                _Span(SPAN_EDGE, f" ──{label}──▶ "),
                _Span(SPAN_TEXT, target),
            ]
        )
    art = MermaidArt(plain=lines, styled=styled, warnings=warnings)
    art.width = max((_display_width(line) for line in lines), default=0)
    return art


def render_mermaid(code: str) -> MermaidArt | None:
    """把 mermaid 代码渲染成终端图；不支持的类型返回 None。"""
    lines = [line for line in code.splitlines()]
    if not lines:
        return None
    directive = lines[0].strip().lower()
    body = lines[1:]
    if directive.startswith(("flowchart", "graph")):
        direction = "td"
        direction_match = re.match(r"^(?:flowchart|graph)\s+([A-Za-z]+)", directive)
        if direction_match:
            direction = direction_match.group(1).lower()
        nodes, edges, warnings = _parse_flowchart(body)
        node_map = {node.id: node for node in nodes}
        # 边引用了未声明节点时自动补节点
        for edge in edges:
            for node_id in (edge.source, edge.target):
                if node_id not in node_map:
                    node_map[node_id] = _Node(id=node_id, label=node_id, shape="box")
        if not node_map:
            return MermaidArt(plain=[], warnings=["Empty diagram"])
        if direction in ("td", "tb", "bt"):
            art = _flowchart_td(node_map, edges)
            if direction == "bt":
                # 自底向上：行序反转（箭头方向近似对齐）。
                art.plain = list(reversed(art.plain))
                art.styled = list(reversed(art.styled))
        else:
            art = _flowchart_lr(node_map, edges)
        art.warnings.extend(warnings)
        return art
    if directive.startswith("sequencediagram"):
        return _sequence(body)
    return None
