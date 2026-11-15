"""Tests of the Vercel path restoration."""

from outlier_detector.serving.vercel import restore_original_path


def test_restore_original_path_uses_the_vercel_parameter() -> None:
    scope = {
        "type": "http",
        "path": "/api/index.py",
        "query_string": b"__vercel_path=predict",
    }
    restored = restore_original_path(scope)
    assert restored["path"] == "/predict"
    assert restored["raw_path"] == b"/predict"
    assert restored["query_string"] == b""


def test_restore_original_path_keeps_other_query_parameters() -> None:
    scope = {
        "type": "http",
        "path": "/api/index.py",
        "query_string": b"__vercel_path=models&limit=5",
    }
    restored = restore_original_path(scope)
    assert restored["path"] == "/models"
    assert restored["query_string"] == b"limit=5"


def test_restore_original_path_keeps_the_scope_without_parameter() -> None:
    scope = {"type": "http", "path": "/health", "query_string": b""}
    assert restore_original_path(scope) is scope
