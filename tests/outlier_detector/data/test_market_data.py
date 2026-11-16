"""Tests of the market data functions, with yfinance mocked."""

import pandas as pd
import pytest

from outlier_detector.data import market_data


class FakeTicker:
    """Stand-in for ``yfinance.Ticker`` with fixed answers.

    Parameters
    ----------
    history_frame : pd.DataFrame
        Frame returned by ``history``.
    earnings_frame : pd.DataFrame
        Frame returned by ``get_earnings_dates``.
    """

    def __init__(
        self,
        history_frame: pd.DataFrame,
        earnings_frame: pd.DataFrame,
    ) -> None:
        self.history_frame = history_frame
        self.earnings_frame = earnings_frame

    def history(self, start: str, auto_adjust: bool) -> pd.DataFrame:
        """Return the fixed price history.

        Parameters
        ----------
        start : str
            Ignored; kept to match yfinance.
        auto_adjust : bool
            Ignored; kept to match yfinance.

        Returns
        -------
        pd.DataFrame
            The fixed price history.
        """
        return self.history_frame

    def get_earnings_dates(self, limit: int) -> pd.DataFrame:
        """Return the fixed earnings frame.

        Parameters
        ----------
        limit : int
            Ignored; kept to match yfinance.

        Returns
        -------
        pd.DataFrame
            The fixed earnings frame.
        """
        return self.earnings_frame


def make_history() -> pd.DataFrame:
    """Build a price history with timezone-aware dates and extra columns.

    Returns
    -------
    pd.DataFrame
        Frame with Open, Close and Volume columns, indexed by aware dates.
    """
    index = pd.date_range("2024-01-01", periods=3, tz="America/Sao_Paulo")
    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [10, 20, 30],
        },
        index=index,
    )


def test_load_prices_keeps_close_and_volume_without_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTicker(make_history(), pd.DataFrame())
    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: fake)
    prices = market_data.load_prices("AAPL", "2024-01-01")
    assert list(prices.columns) == ["Close", "Volume"]
    assert prices.index.tz is None


def test_load_prices_raises_when_there_is_no_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeTicker(pd.DataFrame(), pd.DataFrame())
    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: fake)
    with pytest.raises(ValueError, match="No price data"):
        market_data.load_prices("NOPE", "2024-01-01")


def test_load_benchmark_log_returns_handles_multi_index_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = pd.MultiIndex.from_tuples([("Close", "^BVSP")])
    downloaded = pd.DataFrame(
        [[100.0], [101.0], [102.0]],
        index=pd.date_range("2024-01-01", periods=3),
        columns=columns,
    )
    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda *args, **kwargs: downloaded,
    )
    returns = market_data.load_benchmark_log_returns("^BVSP", "2024-01-01")
    assert returns.iloc[1] == pytest.approx(0.00995, abs=1e-4)
    assert returns.index.tz is None


def test_load_benchmark_raises_when_there_is_no_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        market_data.yf,
        "download",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="No price data for benchmark"):
        market_data.load_benchmark_log_returns("^BVSP", "2024-01-01")


def test_load_earnings_days_returns_normalized_naive_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    earnings = pd.DataFrame(
        index=pd.DatetimeIndex(
            ["2024-02-14 16:00:00-05:00", "2024-05-02 16:00:00-05:00"]
        )
    )
    fake = FakeTicker(pd.DataFrame(), earnings)
    monkeypatch.setattr(market_data.yf, "Ticker", lambda ticker: fake)
    days = market_data.load_earnings_days("AAPL")
    expected = [pd.Timestamp("2024-02-14"), pd.Timestamp("2024-05-02")]
    assert list(days) == expected
    assert days.tz is None
