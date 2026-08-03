#!/usr/bin/env python3
"""模型目录生成系统（对齐 TS scripts/generate-models.ts 的产物消费端）。

策略（见 docs/04_python-implementation-plan/pi-ai/0803 D1）：
Python 不重新抓取上游，直接消费 TS 生成的 JSON 目录作为数据源，
本脚本只做格式转换（TS camelCase → Python snake_case）并输出到
src/pi_ai/models/generated/。

用法：
    python -m pi_ai.scripts.generate_models --source <ts-data-dir|models.json> [--out <dir>]
    python src/pi_ai/scripts/generate_models.py --source <ts-data-dir|models.json>

输入支持：
    - TS --json-only 产物目录（providers/<id>.json + models.json）
    - TS providers/data 目录（扁平的 <id>.json）
    - 单个 models.json（{provider: {modelId: {...}}}）
"""

import argparse
import json
import sys

from datetime import datetime, timezone
from pathlib import Path

# 本文件位于 <repo>/src/pi_ai/scripts/ 或安装后的 pi_ai/scripts/。
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1]

# 允许从仓库任意位置直接运行（未安装时）。
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# 仓库根（默认输出目录用）；安装到 site-packages 时退化为包目录。
if (_SRC.parent / "src" / "pi_ai").is_dir():
    REPO_ROOT = _SRC.parent
else:
    REPO_ROOT = _HERE.parent


def _number(value) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _convert_cost(cost: dict | None) -> dict:
    cost = cost or {}
    tiers = []
    for tier in cost.get("tiers") or []:
        tiers.append(
            {
                "input": _number(tier.get("input")),
                "output": _number(tier.get("output")),
                "cache_read": _number(tier.get("cacheRead")),
                "cache_write": _number(tier.get("cacheWrite")),
                "input_tokens_above": int(tier.get("inputTokensAbove", 0) or 0),
            }
        )
    return {
        "input": _number(cost.get("input")),
        "output": _number(cost.get("output")),
        "cache_read": _number(cost.get("cacheRead")),
        "cache_write": _number(cost.get("cacheWrite")),
        "tiers": tiers,
    }


def convert_ts_model(ts: dict) -> dict:
    """TS Model（camelCase）→ Python Model 字典（snake_case）。"""
    input_modalities = list(ts.get("input") or [])
    return {
        "id": ts["id"],
        "provider": ts.get("provider", ""),
        "api": ts.get("api", ""),
        "name": ts.get("name", ""),
        "input": input_modalities,
        "output": list(ts.get("output") or []),
        "cost": _convert_cost(ts.get("cost")),
        "max_tokens": int(ts.get("maxTokens", 4096) or 4096),
        "base_url": ts.get("baseUrl", "") or "",
        "context_window": int(ts.get("contextWindow", 0) or 0),
        "headers": dict(ts["headers"]) if ts.get("headers") else None,
        "compat": dict(ts["compat"]) if ts.get("compat") else None,
        "thinking_level_map": (
            dict(ts["thinkingLevelMap"]) if ts.get("thinkingLevelMap") else None
        ),
        "reasoning": bool(ts.get("reasoning")),
    }


def load_ts_catalog(source: Path) -> dict[str, dict[str, dict]]:
    """加载 TS JSON 目录 → {provider_id: {model_id: model}}。"""
    if source.is_file():
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid catalog file: {source}")
        return data
    if not source.is_dir():
        raise ValueError(f"Source does not exist: {source}")

    providers_dir = source / "providers"
    candidates: list[Path] = []
    if providers_dir.is_dir():
        candidates.extend(sorted(providers_dir.glob("*.json")))
    else:
        candidates.extend(
            path
            for path in sorted(source.glob("*.json"))
            if path.name not in ("models.json", "providers.json")
        )
    if not candidates:
        raise ValueError(f"No provider JSON files found under {source}")

    catalog: dict[str, dict[str, dict]] = {}
    seen = set()
    for path in candidates:
        if path.name == "providers.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        provider_id = path.stem
        if provider_id in seen:
            continue
        seen.add(provider_id)
        catalog[provider_id] = data
    return catalog


def write_generated(
    catalog: dict[str, dict[str, dict]],
    out_dir: Path,
) -> None:
    """写出 src/pi_ai/models/generated/ 包内容。"""
    providers_dir = out_dir / "providers"
    providers_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    provider_ids = sorted(catalog)
    for provider_id in provider_ids:
        converted = {
            model_id: convert_ts_model(model)
            for model_id, model in sorted(catalog[provider_id].items())
        }
        (providers_dir / f"{provider_id}.json").write_text(
            json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    (out_dir / "__init__.py").write_text(
        _generated_init_template(generated_at, provider_ids),
        encoding="utf-8",
    )


def _generated_init_template(generated_at: str, provider_ids: list[str]) -> str:
    ids_repr = ", ".join(repr(p) for p in provider_ids)
    return f'''"""自动生成的模型目录（由 src/pi_ai/scripts/generate_models.py 生成，勿手改）。"""

import json
from pathlib import Path

from ..models_store import model_from_dict
from ...types import Model

GENERATED_AT = {generated_at!r}
MODEL_PROVIDERS: list[str] = [{ids_repr}]


def load_generated_models() -> dict[str, list[Model]]:
    """读取 providers/*.json，返回 {{provider_id: [Model, ...]}}。"""
    data_dir = Path(__file__).parent / "providers"

    result: dict[str, list[Model]] = {{}}
    for provider_id in MODEL_PROVIDERS:
        path = data_dir / f"{{provider_id}}.json"
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        result[provider_id] = [model_from_dict(m) for m in raw.values()]
    return result


__all__ = ["GENERATED_AT", "MODEL_PROVIDERS", "load_generated_models"]
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="TS JSON 目录或 models.json")
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            REPO_ROOT / "src" / "pi_ai" / "models" / "generated"
            if (_SRC.parent / "src" / "pi_ai").is_dir()
            else _HERE.parent / "models" / "generated"
        ),
        help="输出目录（默认 src/pi_ai/models/generated）",
    )
    args = parser.parse_args(argv)

    catalog = load_ts_catalog(args.source)
    write_generated(catalog, args.out)
    print(
        f"Generated {len(catalog)} providers under {args.out} "
        f"({sum(len(v) for v in catalog.values())} models)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
