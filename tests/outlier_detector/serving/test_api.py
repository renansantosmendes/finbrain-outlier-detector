"""Tests of the HTTP routes, with the registry and market data mocked."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from outlier_detector.serving import api
from outlier_detector.serving.api import (
    configured_tickers,
    create_app,
)
from outlier_detector.serving.model_registry import WandbModelRegistry


@pytest.fixture
def client(loaded_registry: WandbModelRegistry) -> TestClient:
    """Build a test client on a registry with one detector in memory.

    Parameters
    ----------
    loaded_registry : WandbModelRegistry
        Registry holding the AAPL detector.

    Returns
    -------
    TestClient
        Client of the application, with no startup download.
    """
    return TestClient(create_app(registry=loaded_registry, tickers=[]))


def test_health_reports_the_number_of_models(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models_loaded": 1}


def test_models_lists_the_tickers_in_memory(client: TestClient) -> None:
    assert client.get("/models").json() == {"tickers": ["AAPL"]}


def test_predict_returns_the_scored_days(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"ticker": "AAPL", "last_n_days": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert len(body["days"]) == 2
    assert set(body["days"][0]) == {"day", "score", "is_atypical"}


def test_predict_returns_404_for_a_ticker_not_in_memory(
    client: TestClient,
) -> None:
    response = client.post("/predict", json={"ticker": "NOPE"})
    assert response.status_code == 404


def test_predict_rejects_an_unsafe_ticker(client: TestClient) -> None:
    response = client.post("/predict", json={"ticker": "../etc"})
    assert response.status_code == 422


def test_predict_rejects_a_request_without_ticker(client: TestClient) -> None:
    response = client.post("/predict", json={"last_n_days": 3})
    assert response.status_code == 422


def test_configured_tickers_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTLIER_TICKERS", "AAPL, PETR4.SA,")
    assert configured_tickers() == ["AAPL", "PETR4.SA"]


def test_configured_tickers_is_empty_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OUTLIER_TICKERS", raising=False)
    assert configured_tickers() == []


def test_create_app_preloads_the_configured_tickers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested = []

    def fake_load_all(
        self: WandbModelRegistry,
        tickers: list[str],
    ) -> list[str]:
        """Record the tickers requested at startup.

        Parameters
        ----------
        self : WandbModelRegistry
            Registry instance, ignored.
        tickers : list[str]
            Tickers requested at startup.

        Returns
        -------
        list[str]
            Empty list; nothing is actually loaded.
        """
        requested.extend(tickers)
        return []

    monkeypatch.setattr(WandbModelRegistry, "load_all", fake_load_all)
    monkeypatch.setenv("OUTLIER_TICKERS", "AAPL,PETR4.SA")
    registry = WandbModelRegistry(entity="team", download_directory=tmp_path)
    create_app(registry=registry)
    assert requested == ["AAPL", "PETR4.SA"]


def test_build_default_registry_reads_the_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTLIER_WANDB_PROJECT", "my-project")
    monkeypatch.setenv("WANDB_ENTITY", "my-team")
    registry = api.build_default_registry()
    assert registry.project_name == "my-project"
    assert registry.entity == "my-team"


def test_predict_logs_the_served_prediction(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="outlier_detector.serving.api"):
        client.post("/predict", json={"ticker": "AAPL", "last_n_days": 2})
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Prediction served: ticker=AAPL last_n_days=2 days=2" in message
        for message in messages
    )


def test_predict_logs_a_warning_for_a_missing_model(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="outlier_detector.serving.api"):
        client.post("/predict", json={"ticker": "NOPE"})
    warnings = [
        record for record in caplog.records if record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert "ticker=NOPE" in warnings[0].getMessage()


def test_startup_logs_the_models_that_were_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        WandbModelRegistry,
        "load_all",
        lambda self, tickers: ["AAPL"],
    )
    registry = WandbModelRegistry(entity="team", download_directory=tmp_path)
    with caplog.at_level("INFO", logger="outlier_detector.serving.api"):
        create_app(registry=registry, tickers=["AAPL", "NOPE"])
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Startup loaded 1 of 2 model(s): loaded=['AAPL']" in message
        for message in messages
    )
