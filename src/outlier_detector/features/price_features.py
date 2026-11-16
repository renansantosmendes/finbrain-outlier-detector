"""Build the model inputs from prices, benchmark returns and earnings dates."""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret",
    "ret_abs",
    "ret_lag1",
    "vol_20",
    "vol_rel",
    "ret_ab",
    "is_earn",
]
ROLLING_WINDOW = 20


def build_features(
    prices: pd.DataFrame,
    benchmark_log_returns: pd.Series,
    earnings_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Turn daily prices into the features used by the detector.

    The features are the daily log return, its absolute value and its
    lagged value, the 20-day volatility, the volume relative to its 20-day
    average, the return in excess of the benchmark and a flag for the days
    on which the company published its results. Every feature uses only the
    current day or earlier days.

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
    features = pd.DataFrame(index=prices.index)
    features["ret"] = np.log(prices["Close"]).diff()
    features["ret_abs"] = features["ret"].abs()
    features["ret_lag1"] = features["ret"].shift(1)
    features["vol_20"] = features["ret"].rolling(ROLLING_WINDOW).std()
    features["vol_rel"] = prices["Volume"] / (
        prices["Volume"].rolling(ROLLING_WINDOW).mean()
    )
    features["ret_ab"] = features["ret"] - benchmark_log_returns.reindex(
        prices.index
    )
    features["is_earn"] = prices.index.normalize().isin(earnings_days).astype(
        int
    )
    return (
        features[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
