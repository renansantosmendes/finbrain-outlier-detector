"""Build the model inputs from daily prices."""

import numpy as np
import pandas as pd

from outlier_detector.features.indicators import (
    macd_histogram,
    relative_strength_index,
)

FEATURE_COLUMNS = ["log_return", "volume_z_score", "rsi", "macd_normalized"]
VOLUME_WINDOW = 60
MINIMUM_PRICE_ROWS = VOLUME_WINDOW


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Turn daily prices into the features used by the detector.

    The features are the daily log return, the z-score of the log volume
    against its 60-day window, the 14-day RSI and the MACD histogram
    divided by the close price. Rows that lack a full window are removed,
    so the first ``MINIMUM_PRICE_ROWS - 1`` days produce no features.

    Parameters
    ----------
    prices : pd.DataFrame
        Frame with the columns ``Close`` and ``Volume``, sorted by date.

    Returns
    -------
    pd.DataFrame
        Frame with the columns listed in ``FEATURE_COLUMNS``, indexed like
        ``prices``, without missing values.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> days = pd.date_range("2024-01-01", periods=120, freq="B")
    >>> prices = pd.DataFrame(
    ...     {"Close": np.linspace(10, 20, 120), "Volume": 1000.0},
    ...     index=days,
    ... )
    >>> list(build_features(prices).columns)
    ['log_return', 'volume_z_score', 'rsi', 'macd_normalized']
    """
    close = prices["Close"]
    log_volume = np.log1p(prices["Volume"])
    volume_mean = log_volume.rolling(VOLUME_WINDOW).mean()
    volume_std = log_volume.rolling(VOLUME_WINDOW).std()

    features = pd.DataFrame(index=prices.index)
    features["log_return"] = np.log(close).diff()
    features["volume_z_score"] = (log_volume - volume_mean) / volume_std
    features["rsi"] = relative_strength_index(close)
    features["macd_normalized"] = macd_histogram(close) / close
    return features[FEATURE_COLUMNS].dropna()
