"""Tests of the feature builder."""

import numpy as np
import pytest

from outlier_detector.features.price_features import (
    FEATURE_COLUMNS,
    MINIMUM_PRICE_ROWS,
    build_features,
)
from tests.outlier_detector.conftest import make_prices


def test_build_features_returns_the_expected_columns() -> None:
    features = build_features(make_prices(200))
    assert list(features.columns) == FEATURE_COLUMNS


def test_build_features_has_no_missing_values() -> None:
    features = build_features(make_prices(200))
    assert not features.isna().any().any()


def test_build_features_needs_a_full_volume_window() -> None:
    features = build_features(make_prices(MINIMUM_PRICE_ROWS))
    assert len(features) == 1


def test_build_features_is_empty_with_too_few_rows() -> None:
    features = build_features(make_prices(MINIMUM_PRICE_ROWS - 1))
    assert features.empty


def test_build_features_keeps_dates_of_the_prices() -> None:
    prices = make_prices(120)
    features = build_features(prices)
    assert features.index.isin(prices.index).all()
    assert features.index.is_monotonic_increasing


def test_log_return_matches_the_close_ratio() -> None:
    prices = make_prices(120)
    features = build_features(prices)
    day = features.index[-1]
    position = prices.index.get_loc(day)
    close = prices["Close"]
    expected = np.log(close.iloc[position] / close.iloc[position - 1])
    assert features.loc[day, "log_return"] == pytest.approx(expected)
