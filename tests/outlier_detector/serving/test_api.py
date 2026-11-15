"""Tests of the HTTP routes, using the FastAPI test client."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from outlier_detector.serving.api import create_app, default_model_directory
from tests.outlier_detector.serving.test_service import (
    price_records,
    train_detector,
)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Build a test client on a model folder with one trained ticker.

    Parameters
    ----------
    tmp_path : Path
        Pytest folder for this test.

    Returns
    -------
    TestClient
        Client of the application.
    """
    train_detector(tmp_path, "AAPL")
    return TestClient(create_app(tmp_path))


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_lists_trained_tickers(client: TestClient) -> None:
    response = client.get("/models")
    assert response.json() == {"tickers": ["AAPL"]}


def test_predict_returns_scored_days(client: TestClient) -> None:
    body = {
        "ticker": "AAPL",
        "prices": [
            {
                "day": record.day.isoformat(),
                "close": record.close,
                "volume": record.volume,
            }
            for record in price_records(90)
        ],
        "last_n_days": 2,
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    days = response.json()["days"]
    assert len(days) == 2
    assert set(days[0]) == {"day", "score", "is_atypical"}


def test_predict_returns_404_for_unknown_ticker(client: TestClient) -> None:
    body = {
        "ticker": "NOPE",
        "prices": [
            {
                "day": record.day.isoformat(),
                "close": record.close,
                "volume": record.volume,
            }
            for record in price_records(70)
        ],
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 404


def test_predict_returns_422_for_invalid_prices(client: TestClient) -> None:
    body = {
        "ticker": "AAPL",
        "prices": [{"day": "2024-01-02", "close": -1, "volume": 1}],
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 422


def test_default_model_directory_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OUTLIER_MODEL_DIRECTORY", str(tmp_path))
    assert default_model_directory() == tmp_path


def test_default_model_directory_falls_back_to_project_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OUTLIER_MODEL_DIRECTORY", raising=False)
    assert default_model_directory().name == "models"
