"""Tests of the scoring service, with market data and the registry mocked."""

from datetime import date
import pytest

from outlier_detector.serving.model_registry import (
    ModelNotFoundError,
    WandbModelRegistry,
)
from outlier_detector.serving.schemas import PredictRequest
from outlier_detector.serving.service import detect_atypical_days
from tests.outlier_detector.conftest import (
    make_market_inputs,
    patch_service_market_data,
)


def test_detect_atypical_days_returns_the_last_days(
    loaded_registry: WandbModelRegistry,
) -> None:
    request = PredictRequest(ticker="AAPL", last_n_days=3)
    response = detect_atypical_days(request, loaded_registry)
    assert response.ticker == "AAPL"
    assert response.model_trained_at == "2026-01-01T00:00:00+00:00"
    assert len(response.days) == 3
    assert [day.day for day in response.days] == sorted(
        day.day for day in response.days
    )


def test_detect_atypical_days_returns_scores_and_labels(
    loaded_registry: WandbModelRegistry,
) -> None:
    request = PredictRequest(ticker="AAPL", last_n_days=2)
    response = detect_atypical_days(request, loaded_registry)
    assert all(isinstance(day.is_atypical, bool) for day in response.days)
    assert all(isinstance(day.score, float) for day in response.days)


def test_detect_atypical_days_raises_for_a_ticker_not_in_memory(
    loaded_registry: WandbModelRegistry,
) -> None:
    request = PredictRequest(ticker="NOPE")
    with pytest.raises(ModelNotFoundError):
        detect_atypical_days(request, loaded_registry)


def test_detect_atypical_days_raises_when_there_are_no_features(
    loaded_registry: WandbModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    short_prices, benchmark, earnings = make_market_inputs(10)
    patch_service_market_data(monkeypatch, short_prices, benchmark, earnings)
    request = PredictRequest(ticker="AAPL")
    with pytest.raises(ValueError, match="Not enough market data"):
        detect_atypical_days(request, loaded_registry, today=date(2026, 1, 1))
