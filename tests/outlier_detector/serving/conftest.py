"""Fixtures of the serving tests."""

from pathlib import Path

import pytest

from outlier_detector.features.price_features import (
    FEATURE_COLUMNS,
    build_features,
)
from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)
from outlier_detector.serving.model_registry import (
    LoadedModel,
    WandbModelRegistry,
)
from tests.outlier_detector.conftest import (
    make_market_inputs,
    patch_service_market_data,
)


@pytest.fixture
def loaded_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> WandbModelRegistry:
    """Build a registry with one detector in memory and mocked market data.

    Parameters
    ----------
    tmp_path : Path
        Pytest folder used by the registry.
    monkeypatch : pytest.MonkeyPatch
        Pytest helper used to replace the market data functions.

    Returns
    -------
    WandbModelRegistry
        Registry holding a detector fitted on the synthetic features.
    """
    prices, benchmark, earnings = make_market_inputs(400)
    patch_service_market_data(monkeypatch, prices, benchmark, earnings)

    registry = WandbModelRegistry(
        entity="team",
        download_directory=tmp_path,
        env_file_path=str(tmp_path / "missing.env"),
    )
    features = build_features(prices, benchmark, earnings)
    detector = IsolationForestDetector().fit(features[FEATURE_COLUMNS])
    registry.models["AAPL"] = LoadedModel(
        ticker="AAPL",
        detector=detector,
        metadata={"trained_at": "2026-01-01T00:00:00+00:00"},
    )
    return registry
