"""Artifact 持久化（对齐 TS vitest-evals/artifacts.ts）。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .harness import JsonValue

PI_SESSION_SNAPSHOT_ARTIFACT = "piSessionJsonl"
PI_EVAL_SOURCES_ARTIFACT = "piEvalSources"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def persist_eval_artifact_references(
    artifacts: dict[str, JsonValue],
    run_id: str,
    artifact_dir: Path,
) -> list[dict[str, str]]:
    """把 session 快照与 eval 源文件写入 artifact 目录，返回引用列表。

    目录布局（对齐 TS）：sessions/<sha256(runId)>/session.jsonl 与
    sources/<sha256(runId)>/<name>。
    """
    references: list[dict[str, str]] = []
    session = artifacts.get(PI_SESSION_SNAPSHOT_ARTIFACT)
    if isinstance(session, str):
        directory = artifact_dir / "sessions" / _sha256(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "session.jsonl"
        path.write_text(session, encoding="utf-8")
        references.append({"name": "session.jsonl", "path": str(path.relative_to(artifact_dir))})

    sources = artifacts.get(PI_EVAL_SOURCES_ARTIFACT)
    if isinstance(sources, list):
        directory = artifact_dir / "sources" / _sha256(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        for source in sources:
            if not isinstance(source, dict):
                continue
            name = source.get("name")
            body = source.get("body")
            if not isinstance(name, str) or not isinstance(body, str):
                continue
            if Path(name).name != name:
                raise TypeError(f"Invalid eval artifact name: {name}")
            path = directory / name
            path.write_text(body, encoding="utf-8")
            references.append({"name": name, "path": str(path.relative_to(artifact_dir))})
    return references


__all__ = [
    "PI_EVAL_SOURCES_ARTIFACT",
    "PI_SESSION_SNAPSHOT_ARTIFACT",
    "persist_eval_artifact_references",
]
