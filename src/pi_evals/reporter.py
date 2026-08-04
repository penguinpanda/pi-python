"""评测结果汇总（对齐 TS vitest-evals/reporter + summary）。"""

from __future__ import annotations

from .harness import EvalResult


def report_results(results: list[EvalResult], *, title: str = "Eval summary") -> str:
    """把多次 eval 运行渲染为文本汇总表。"""
    lines = [title, "=" * len(title), ""]
    if not results:
        lines.append("(no results)")
        return "\n".join(lines)
    lines.append(f"{'name':<28} {'status':<8} {'duration_ms':<12} {'tokens':<8}")
    lines.append("-" * 60)
    for index, result in enumerate(results, start=1):
        status = "FAIL" if result.errors else "ok"
        tokens = result.usage.get("totalTokens", 0)
        lines.append(f"{index:<28} {status:<8} {result.duration_ms:<12} {tokens:<8}")
    failed = [index for index, result in enumerate(results, 1) if result.errors]
    if failed:
        lines.append("")
        lines.append(f"Failed runs: {failed}")
        for index in failed:
            result = results[index - 1]
            for error in result.errors:
                lines.append(f"  [{index}] {error}")
    return "\n".join(lines)


__all__ = ["report_results"]
