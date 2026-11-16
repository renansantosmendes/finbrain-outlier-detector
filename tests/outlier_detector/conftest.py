"""Shared synthetic inputs of the outlier detector tests."""

import numpy as np
import pandas as pd
import pytest

from outlier_detector.serving import service


def make_market_inputs(
    days: int,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Build synthetic prices, benchmark returns and earnings dates.

    Parameters
    ----------
    days : int
        Number of trading days.
    seed : int, optional
        Seed of the random generator. The default is 0.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]
        Prices with the columns ``Close`` and ``Volume``, the benchmark log
        returns on the same dates, and a few earnings dates.
    """
    generator = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=days)
    close = 20 * np.exp(np.cumsum(generator.normal(0.0005, 0.01, size=days)))
    prices = pd.DataFrame(
        {
            "Close": close,
            "Volume": generator.integers(
                1_000, 5_000, size=days
            ).astype(float),
        },
        index=index,
    )
    benchmark = pd.Series(
        generator.normal(0.0002, 0.008, size=days),
        index=index,
    )
    earnings = pd.DatetimeIndex(
        [index[days // 6], index[days // 2], index[days - 5]]
    )
    return prices, benchmark, earnings


def patch_service_market_data(
    monkeypatch: pytest.MonkeyPatch,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    earnings: pd.DatetimeIndex,
) -> None:
    """Replace the market data downloads of the scoring service.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest helper used to replace the functions.
    prices : pd.DataFrame
        Prices returned for any ticker and start date.
    benchmark : pd.Series
        Benchmark log returns returned for any symbol and start date.
    earnings : pd.DatetimeIndex
        Earnings dates returned for any ticker.
    """
    monkeypatch.setattr(service, "load_prices", lambda ticker, start: prices)
    monkeypatch.setattr(
        service,
        "load_benchmark_log_returns",
        lambda symbol, start: benchmark,
    )
    monkeypatch.setattr(
        service,
        "load_earnings_days",
        lambda ticker: earnings,
    )
