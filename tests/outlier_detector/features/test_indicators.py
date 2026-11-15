"""Tests of the technical indicators."""

import numpy as np
import pandas as pd

from outlier_detector.features.indicators import (
    macd_histogram,
    relative_strength_index,
)


def test_rsi_is_100_when_price_only_rises() -> None:
    close = pd.Series(np.arange(1.0, 31.0))
    rsi = relative_strength_index(close, window=5)
    assert rsi.dropna().eq(100.0).all()


def test_rsi_is_nan_before_the_window_is_full() -> None:
    close = pd.Series(np.arange(1.0, 31.0))
    rsi = relative_strength_index(close, window=5)
    assert rsi.iloc[:5].isna().all()
    assert rsi.iloc[5:].notna().all()


def test_rsi_stays_between_0_and_100() -> None:
    generator = np.random.default_rng(3)
    close = pd.Series(50 + np.cumsum(generator.normal(size=200)))
    rsi = relative_strength_index(close).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_macd_histogram_is_zero_for_constant_prices() -> None:
    close = pd.Series(np.full(100, 10.0))
    histogram = macd_histogram(close)
    assert np.allclose(histogram.to_numpy(), 0.0)


def test_macd_histogram_keeps_the_index() -> None:
    index = pd.date_range("2024-01-01", periods=50, freq="B")
    close = pd.Series(np.linspace(10, 20, 50), index=index)
    assert macd_histogram(close).index.equals(index)
