"""Tests of the feature builder."""

import numpy as np
import pytest

from outlier_detector.features.price_features import (
    FEATURE_COLUMNS,
    build_features,
)
from tests.outlier_detector.conftest import make_market_inputs


def test_build_features_returns_the_expected_columns() -> None:
    features = build_features(*make_market_inputs(200))
    assert list(features.columns) == FEATURE_COLUMNS


def test_build_features_has_no_missing_values() -> None:
    features = build_features(*make_market_inputs(200))
    assert not features.isna().any().any()


def test_build_features_drops_the_warm_up_days() -> None:
    prices, _, _ = make_market_inputs(200)
    features = build_features(*make_market_inputs(200))
    assert features.index[0] == prices.index[20]


def test_is_earn_marks_the_earnings_days() -> None:
    prices, benchmark, earnings = make_market_inputs(200)
    features = build_features(prices, benchmark, earnings)
    assert features.loc[earnings[0], "is_earn"] == 1
    assert features["is_earn"].sum() == len(earnings)


def test_ret_ab_is_the_return_minus_the_benchmark() -> None:
    prices, benchmark, earnings = make_market_inputs(200)
    features = build_features(prices, benchmark, earnings)
    day = features.index[-1]
    expected_ret = np.log(prices["Close"]).diff().loc[day]
    assert features.loc[day, "ret"] == pytest.approx(expected_ret)
    assert features.loc[day, "ret_ab"] == pytest.approx(
        expected_ret - benchmark.loc[day]
    )


def test_ret_lag1_is_the_previous_day_return() -> None:
    prices, benchmark, earnings = make_market_inputs(200)
    features = build_features(prices, benchmark, earnings)
    day = features.index[-1]
    previous_day = features.index[-2]
    assert features.loc[day, "ret_lag1"] == pytest.approx(
        features.loc[previous_day, "ret"]
    )
