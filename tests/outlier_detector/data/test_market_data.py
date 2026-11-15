"""Tests of the price download, with yfinance mocked."""

import pandas as pd
import pytest

from outlier_detector.data import market_data


def test_load_prices_keeps_close_and_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.DataFrame(
        {"Close": [1.0, 2.0], "Volume": [10.0, 20.0], "Open": [0.5, 1.5]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda *args, **kwargs: raw,
    )
    prices = market_data.load_prices("AAPL")
    assert list(prices.columns) == ["Close", "Volume"]
    assert len(prices) == 2


def test_load_prices_flattens_multi_index_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("Volume", "AAPL")]
    )
    raw = pd.DataFrame([[1.0, 10.0], [2.0, 20.0]], columns=columns)
    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda *args, **kwargs: raw,
    )
    prices = market_data.load_prices("AAPL")
    assert list(prices.columns) == ["Close", "Volume"]


def test_load_prices_raises_when_there_is_no_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda *args, **kwargs: pd.DataFrame(columns=["Close", "Volume"]),
    )
    with pytest.raises(ValueError, match="No price data"):
        market_data.load_prices("NOPE")
