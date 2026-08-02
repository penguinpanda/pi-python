"""flatten_model_catalog 测试。"""

from pi_ai.models.model_catalog import flatten_model_catalog
from pi_ai.types import Model


def _model(model_id: str) -> Model:
    return Model(id=model_id, provider="p", api="openai-completions")


def test_flatten_model_catalog():
    groups = {
        "openai-completions": {"a": _model("a"), "b": _model("b")},
        "openai-responses": {"c": _model("c")},
    }
    flat = flatten_model_catalog("p", groups)
    assert set(flat) == {"a", "b", "c"}
    assert flat["c"].provider == "p"


def test_flatten_model_catalog_empty():
    assert flatten_model_catalog("p", {}) == {}
