"""LaTeX 预处理围栏感知回归测试。"""

from __future__ import annotations

from pi_tui.engine.markdown_render import INLINE_LATEX_START, _preprocess_latex


def test_latex_preprocess_skips_fenced_code() -> None:
    """围栏代码块内的 $ 不被 LaTeX 占位符改写。"""
    source = "```sh\necho $HOME $a$\n```\nmath: $x$"
    processed, _latex_map, _block_ids = _preprocess_latex(source)
    assert "echo $HOME $a$" in processed
    assert INLINE_LATEX_START in processed


def test_latex_block_outside_fence_still_rendered() -> None:
    """围栏外的块级 LaTeX 仍被替换且行结构保持。"""
    source = "before\n$$x^2$$\nafter"
    processed, latex_map, _block_ids = _preprocess_latex(source)
    assert len(latex_map) == 1
    assert processed.count("\n") == 2
