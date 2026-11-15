"""Tests of the training command, with yfinance mocked."""

from pathlib import Path

import pytest

from outlier_detector.models.artifact_store import load_metadata, load_model
from outlier_detector.training import train
from tests.outlier_detector.conftest import make_prices


@pytest.fixture
def mocked_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the yfinance download with synthetic prices.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest helper used to replace ``load_prices``.
    """
    monkeypatch.setattr(
        train,
        "load_prices",
        lambda ticker, period: make_prices(400),
    )


def test_train_ticker_stores_model_and_metadata(
    tmp_path: Path,
    mocked_prices: None,
) -> None:
    metadata = train.train_ticker(
        ticker="AAPL",
        period="3y",
        model_directory=tmp_path,
        contamination=0.02,
        random_seed=42,
    )
    assert metadata["ticker"] == "AAPL"
    assert metadata["training_rows"] == 341
    assert load_metadata(tmp_path, "AAPL")["ticker"] == "AAPL"
    assert load_model(tmp_path, "AAPL").feature_columns


def test_train_ticker_reports_the_walk_forward_share(
    tmp_path: Path,
    mocked_prices: None,
) -> None:
    metadata = train.train_ticker(
        ticker="AAPL",
        period="3y",
        model_directory=tmp_path,
        contamination=0.02,
        random_seed=42,
    )
    share = metadata["walk_forward_flagged_share"]
    assert 0.0 <= share <= 1.0


def test_main_trains_every_ticker(
    tmp_path: Path,
    mocked_prices: None,
) -> None:
    exit_code = train.main(
        [
            "--tickers",
            "AAPL, MSFT",
            "--model-directory",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "AAPL.json").is_file()
    assert (tmp_path / "MSFT.json").is_file()


def test_main_returns_1_when_a_ticker_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_download(ticker: str, period: str) -> None:
        raise ValueError(f"No price data for ticker {ticker!r}.")

    monkeypatch.setattr(train, "load_prices", failing_download)
    exit_code = train.main(
        ["--tickers", "NOPE", "--model-directory", str(tmp_path)]
    )
    assert exit_code == 1


def test_parse_arguments_splits_the_tickers() -> None:
    options = train.parse_arguments(["--tickers", "AAPL,PETR4.SA,,"])
    assert options.tickers == ["AAPL", "PETR4.SA"]
    assert options.period == "3y"


def test_parse_arguments_rejects_an_empty_ticker_list() -> None:
    with pytest.raises(SystemExit):
        train.parse_arguments(["--tickers", " , "])
