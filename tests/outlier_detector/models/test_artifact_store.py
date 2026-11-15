"""Tests of the model artifact store."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from outlier_detector.models.artifact_store import (
    ModelNotFoundError,
    list_trained_tickers,
    load_metadata,
    load_model,
    model_path,
    safe_file_stem,
    save_model,
)
from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)


def fitted_detector() -> IsolationForestDetector:
    """Fit a small detector on random rows.

    Returns
    -------
    IsolationForestDetector
        Fitted detector with the columns ``a`` and ``b``.
    """
    generator = np.random.default_rng(2)
    training = pd.DataFrame(
        generator.normal(size=(100, 2)),
        columns=["a", "b"],
    )
    return IsolationForestDetector().fit(training)


def test_safe_file_stem_replaces_unsafe_characters() -> None:
    assert safe_file_stem("PETR4.SA") == "PETR4_SA"
    assert safe_file_stem("../secret") == "___secret"


def test_model_path_stays_inside_the_model_folder(tmp_path: Path) -> None:
    path = model_path(tmp_path, "../../etc/passwd")
    assert path.parent == tmp_path


def test_save_model_writes_detector_and_metadata(tmp_path: Path) -> None:
    save_model(tmp_path, "AAPL", fitted_detector(), {"ticker": "AAPL"})
    assert (tmp_path / "AAPL.joblib").is_file()
    assert load_metadata(tmp_path, "AAPL") == {"ticker": "AAPL"}


def test_load_model_returns_the_saved_detector(tmp_path: Path) -> None:
    save_model(tmp_path, "AAPL", fitted_detector(), {"ticker": "AAPL"})
    detector = load_model(tmp_path, "AAPL")
    assert detector.feature_columns == ["a", "b"]


def test_load_model_raises_for_unknown_ticker(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path, "UNKNOWN")


def test_load_metadata_raises_for_unknown_ticker(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        load_metadata(tmp_path, "UNKNOWN")


def test_list_trained_tickers_is_sorted(tmp_path: Path) -> None:
    save_model(
        tmp_path,
        "PETR4.SA",
        fitted_detector(),
        {"ticker": "PETR4.SA"},
    )
    save_model(tmp_path, "AAPL", fitted_detector(), {"ticker": "AAPL"})
    assert list_trained_tickers(tmp_path) == ["AAPL", "PETR4.SA"]


def test_list_trained_tickers_is_empty_without_folder(tmp_path: Path) -> None:
    assert list_trained_tickers(tmp_path / "missing") == []
