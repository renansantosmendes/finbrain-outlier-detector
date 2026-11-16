"""Build the model inputs from prices, benchmark returns and earnings dates."""

import numpy as np
import pandas as pd

LOG_RETURN = "log_return"
ABSOLUTE_LOG_RETURN = "absolute_log_return"
PREVIOUS_LOG_RETURN = "previous_log_return"
VOLATILITY_20_DAYS = "volatility_20_days"
RELATIVE_VOLUME_20_DAYS = "relative_volume_20_days"
EXCESS_LOG_RETURN = "excess_log_return"
IS_EARNINGS_DAY = "is_earnings_day"

FEATURE_COLUMNS = [
    LOG_RETURN,
    ABSOLUTE_LOG_RETURN,
    PREVIOUS_LOG_RETURN,
    VOLATILITY_20_DAYS,
    RELATIVE_VOLUME_20_DAYS,
    EXCESS_LOG_RETURN,
    IS_EARNINGS_DAY,
]
ROLLING_WINDOW_DAYS = 20


def build_features(
    prices: pd.DataFrame,
    benchmark_log_returns: pd.Series,
    earnings_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Turn daily prices into the features used by the detector.

    The features are the daily log return and its absolute value, the
    previous day's log return, the 20-day volatility, the volume relative to
    its 20-day average, the log return in excess of the benchmark and a flag
    for the days on which the company published its results. Every feature
    uses only the current day or earlier days.

    Parameters
    ----------
    prices : pd.DataFrame
        Frame indexed by date with the columns ``Close`` and ``Volume``.
    benchmark_log_returns : pd.Series
        Daily log returns of the benchmark, indexed by date.
    earnings_days : pd.DatetimeIndex
        Normalized dates of the earnings publications.

    Returns
    -------
    pd.DataFrame
        Frame with the columns listed in ``FEATURE_COLUMNS``, indexed like
        ``prices``, without missing values.

    Example
    -------
    >>> features = build_features(
    ...     prices, benchmark_returns, earnings
    ... )  # doctest: +SKIP
    """
    close_prices = prices["Close"]
    traded_volume = prices["Volume"]
    log_returns = np.log(close_prices).diff()
    average_volume = traded_volume.rolling(ROLLING_WINDOW_DAYS).mean()
    benchmark_on_price_days = benchmark_log_returns.reindex(prices.index)

    features = pd.DataFrame(index=prices.index)
    features[LOG_RETURN] = log_returns
    features[ABSOLUTE_LOG_RETURN] = log_returns.abs()
    features[PREVIOUS_LOG_RETURN] = log_returns.shift(1)
    features[VOLATILITY_20_DAYS] = log_returns.rolling(
        ROLLING_WINDOW_DAYS
    ).std()
    features[RELATIVE_VOLUME_20_DAYS] = traded_volume / average_volume
    features[EXCESS_LOG_RETURN] = log_returns - benchmark_on_price_days
    features[IS_EARNINGS_DAY] = (
        prices.index.normalize().isin(earnings_days).astype(int)
    )
    return (
        features[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
