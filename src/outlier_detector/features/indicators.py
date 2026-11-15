"""Technical indicators computed from a price series."""

import pandas as pd


def relative_strength_index(
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Compute the relative strength index (RSI) of a close series.

    The RSI compares the average gain with the average loss over a
    rolling window of daily changes. Values close to 100 mean the price
    rose almost every day in the window; values close to 0 mean it fell.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    window : int, optional
        Number of daily changes in the rolling average. The default is 14.

    Returns
    -------
    pd.Series
        RSI values on the same index as ``close``. The first ``window``
        values are NaN.

    Example
    -------
    >>> import pandas as pd
    >>> prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> relative_strength_index(prices, window=2).iloc[-1]
    100.0
    """
    daily_change = close.diff()
    average_gain = daily_change.clip(lower=0).rolling(window).mean()
    average_loss = (-daily_change.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + average_gain / average_loss)


def macd_histogram(
    close: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.Series:
    """Compute the MACD histogram of a close series.

    The MACD line is the difference between two exponential moving
    averages. The histogram is the MACD line minus its own signal line.

    Parameters
    ----------
    close : pd.Series
        Daily closing prices.
    fast_span : int, optional
        Span of the fast exponential moving average. The default is 12.
    slow_span : int, optional
        Span of the slow exponential moving average. The default is 26.
    signal_span : int, optional
        Span of the signal line. The default is 9.

    Returns
    -------
    pd.Series
        Histogram values on the same index as ``close``.
    """
    macd_line = (
        close.ewm(span=fast_span, adjust=False).mean()
        - close.ewm(span=slow_span, adjust=False).mean()
    )
    signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()
    return macd_line - signal_line
