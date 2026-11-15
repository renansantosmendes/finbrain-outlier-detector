"""Tests of the scoring service and the request schema."""

from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from outlier_detector.models.artifact_store import (
    ModelNotFoundError,
    save_model,
)
from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)
from outlier_detector.features.price_features import (
    FEATURE_COLUMNS,
    build_features,
)
from outlier_detector.serving.schemas import PredictRequest, PriceRecord
from outlier_detector.serving.service import detect_atypical_days
from tests.outlier_detector.conftest import make_prices


def price_records(days: int) -> list[PriceRecord]:
    """Build price records from synthetic prices.

    Parameters
    ----------
    days : int
        Number of trading days.

    Returns
    -------
    list[PriceRecord]
        Records with consecutive business days, oldest first.
    """
    prices = make_prices(days)
    return [
        PriceRecord(
            day=day.date(),
            close=float(row.Close),
            volume=float(row.Volume),
        )
        for day, row in zip(prices.index, prices.itertuples())
    ]


def train_detector(model_directory: Path, ticker: str) -> None:
    """Fit and store a detector on the features of synthetic prices.

    Parameters
    ----------
    model_directory : Path
        Folder where the detector is stored.
    ticker : str
        Ticker the detector belongs to.
    """
    features = build_features(make_prices(400))
    detector = IsolationForestDetector().fit(features[FEATURE_COLUMNS])
    save_model(
        model_directory,
        ticker,
        detector,
        {"ticker": ticker, "trained_at": "2026-01-01T00:00:00+00:00"},
    )


def test_detect_atypical_days_returns_the_last_days(tmp_path: Path) -> None:
    train_detector(tmp_path, "AAPL")
    request = PredictRequest(
        ticker="AAPL",
        prices=price_records(120),
        last_n_days=3,
    )
    response = detect_atypical_days(tmp_path, request)
    assert response.ticker == "AAPL"
    assert response.model_trained_at == "2026-01-01T00:00:00+00:00"
    assert len(response.days) == 3
    assert response.days[-1].day == request.prices[-1].day


def test_detect_atypical_days_raises_for_unknown_ticker(
    tmp_path: Path,
) -> None:
    request = PredictRequest(ticker="NOPE", prices=price_records(120))
    with pytest.raises(ModelNotFoundError):
        detect_atypical_days(tmp_path, request)


def test_predict_request_rejects_unsorted_dates() -> None:
    records = price_records(70)
    records[10], records[11] = records[11], records[10]
    with pytest.raises(ValidationError, match="strictly increasing"):
        PredictRequest(ticker="AAPL", prices=records)


def test_predict_request_rejects_short_history() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(ticker="AAPL", prices=price_records(10))


def test_predict_request_rejects_unsafe_ticker() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(ticker="../etc", prices=price_records(70))


def test_price_record_rejects_non_positive_close() -> None:
    with pytest.raises(ValidationError):
        PriceRecord(day=date(2024, 1, 2), close=0.0, volume=1.0)


def test_price_record_accepts_iso_dates() -> None:
    record = PriceRecord(
        day=date(2024, 1, 2) + timedelta(days=0),
        close=1.0,
        volume=0.0,
    )
    assert record.day == date(2024, 1, 2)
