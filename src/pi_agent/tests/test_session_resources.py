"""Session 资源清理测试。"""

import pytest

from pi_agent.session_resources import (
    CleanupError,
    cleanup_session_resources,
    register_session_resource_cleanup,
)


def test_register_and_cleanup():
    called: list[str | None] = []
    unregister = register_session_resource_cleanup(lambda sid: called.append(sid))
    try:
        cleanup_session_resources("s-1")
        assert called == ["s-1"]
        unregister()
        cleanup_session_resources("s-2")
        assert called == ["s-1"]
    finally:
        unregister()


def test_cleanup_error_aggregates():
    def bad(_sid):
        raise RuntimeError("boom")

    called = []
    unregister_bad = register_session_resource_cleanup(bad)
    unregister_good = register_session_resource_cleanup(lambda sid: called.append(sid))
    try:
        with pytest.raises(CleanupError) as excinfo:
            cleanup_session_resources("s-1")
        assert len(excinfo.value.errors) == 1
        # 失败不阻断其它清理
        assert called == ["s-1"]
    finally:
        unregister_bad()
        unregister_good()


def test_cleanup_empty_registry():
    cleanup_session_resources("s-1")  # 不抛
