"""Download daily prices of a ticker with yfinance."""

import pandas as pd
import yfinance as yf

PRICE_COLUMNS = ["Close", "Volume"]


def load_prices(
    ticker: str,
    period: str = "3y",
) -> pd.DataFrame:
    """Download the daily close and volume of a ticker.

    Prices are adjusted for splits and dividends. Rows with missing
    values are removed.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol, for example ``PETR4.SA`` or ``AAPL``.
    period : str, optional
        Period accepted by ``yfinance``, for example ``1y``, ``3y`` or
        ``max``. The default is ``3y``.

    Returns
    -------
    pd.DataFrame
        Frame indexed by date with the columns ``Close`` and ``Volume``.

    Raises
    ------
    ValueError
        If yfinance returns no rows for the ticker and period.

    Example
    -------
    >>> prices = load_prices("AAPL", period="1y")  # doctest: +SKIP
    >>> list(prices.columns)  # doctest: +SKIP
    ['Close', 'Volume']
    """
    raw_prices = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw_prices.columns, pd.MultiIndex):
        raw_prices.columns = raw_prices.columns.get_level_values(0)
    prices = raw_prices[PRICE_COLUMNS].dropna()
    if prices.empty:
        raise ValueError(f"No price data for ticker {ticker!r}.")
    return prices
