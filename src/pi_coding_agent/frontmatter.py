"""YAML frontmatter 解析（完整 YAML，对齐 TS utils/frontmatter.ts）。"""

from __future__ import annotations

from typing import Any

import yaml


def parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """解析 `---` 包裹的 frontmatter；非法 YAML 抛异常（对齐 TS parseFrontmatter）。"""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.startswith("---"):
        return {}, normalized
    end_marker = "\n---"
    end_index = normalized.find(end_marker, 3)
    if end_index == -1:
        return {}, normalized
    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()
    parsed = yaml.safe_load(yaml_string)
    if parsed is None or not isinstance(parsed, dict):
        frontmatter: dict[str, Any] = {}
    else:
        frontmatter = dict(parsed)
    return frontmatter, body


def strip_frontmatter(content: str) -> str:
    """移除 frontmatter 并返回正文（技能展开用）。"""
    _frontmatter, body = parse_frontmatter(content)
    return body


__all__ = ["parse_frontmatter", "strip_frontmatter"]
