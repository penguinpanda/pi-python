"""evals — pytest 评测 harness（路线图 P2-4）。

pi-harness → PiCodingAgentHarness（faux provider 默认，支持 PI_PROVIDER/
PI_MODEL 选择真实模型）；smoke / extensions 两类 eval；reporter 输出汇总。
"""

from .harness import EvalResult, PiCodingAgentHarness, resolve_model_selection
from .reporter import report_results

__all__ = [
    "EvalResult",
    "PiCodingAgentHarness",
    "resolve_model_selection",
    "report_results",
]
