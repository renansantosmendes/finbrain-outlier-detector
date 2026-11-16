"""Download prices, benchmark returns and earnings dates with yfinance."""

import numpy as np
import pandas as pd
import yfinance as yf

PRICE_COLUMNS = ["Close", "Volume"]
DEFAULT_BENCHMARK = "^BVSP"
EARNINGS_LIMIT = 60


def load_prices(
    ticker: str,
    start: str,
) -> pd.DataFrame:
    """Download the adjusted close and volume of a ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol, for example ``PETR4.SA`` or ``AAPL``.
    start : str
        First date of the history, in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.DataFrame
        Frame indexed by date, without timezone, with the columns ``Close``
        and ``Volume``.

    Raises
    ------
    ValueError
        If yfinance returns no rows for the ticker.

    Example
    -------
    >>> prices = load_prices("AAPL", "2024-01-01")  # doctest: +SKIP
    """
    history = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if history.empty:
        raise ValueError(f"No price data for ticker {ticker!r}.")
    history.index = history.index.tz_localize(None)
    return history[PRICE_COLUMNS].copy()


def load_benchmark_log_returns(
    benchmark: str,
    start: str,
) -> pd.Series:
    """Download the daily log returns of a benchmark index.

    Parameters
    ----------
    benchmark : str
        Yahoo Finance symbol of the index, for example ``^BVSP``.
    start : str
        First date of the history, in ``YYYY-MM-DD`` format.

    Returns
    -------
    pd.Series
        Daily log returns indexed by date, without timezone.

    Raises
    ------
    ValueError
        If yfinance returns no rows for the benchmark.

    Example
    -------
    >>> returns = load_benchmark_log_returns(
    ...     "^BVSP", "2024-01-01"
    ... )  # doctest: +SKIP
    """
    downloaded = yf.download(
        benchmark,
        start=start,
        auto_adjust=True,
        progress=False,
    )
    if downloaded.empty:
        raise ValueError(f"No price data for benchmark {benchmark!r}.")
    close = downloaded["Close"].squeeze()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    return np.log(close).diff()


def load_earnings_days(
    ticker: str,
    limit: int = EARNINGS_LIMIT,
) -> pd.DatetimeIndex:
    """Return the dates on which a company published its results.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol of the company.
    limit : int, optional
        Maximum number of earnings dates requested from yfinance. The
        default is 60.

    Returns
    -------
    pd.DatetimeIndex
        Normalized dates, without timezone.
    """
    earnings = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    return pd.DatetimeIndex(earnings.index).tz_localize(None).normalize()
