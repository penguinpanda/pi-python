"""YAML frontmatter 解析（无 pyyaml 依赖的 key: value 子集）。"""

from __future__ import annotations

from typing import Any


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """解析 `---` 包裹的 frontmatter，返回 (frontmatter, body)。"""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_marker = "\n---"
    end_index = normalized.find(end_marker, 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    frontmatter: dict[str, Any] = {}
    for line in yaml_string.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        frontmatter[key] = value
    return frontmatter, body


def strip_frontmatter(content: str) -> str:
    """移除 frontmatter 并返回正文（技能展开用）。"""
    _frontmatter, body = parse_frontmatter(content)
    return body


__all__ = ["parse_frontmatter", "strip_frontmatter"]
