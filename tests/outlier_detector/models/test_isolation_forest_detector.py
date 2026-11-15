"""Tests of the Isolation Forest detector."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)


def make_training_frame() -> pd.DataFrame:
    """Build normal rows with two columns.

    Returns
    -------
    pd.DataFrame
        300 rows drawn from a standard normal distribution.
    """
    generator = np.random.default_rng(0)
    return pd.DataFrame(generator.normal(size=(300, 2)), columns=["a", "b"])


def test_fit_records_the_feature_columns() -> None:
    detector = IsolationForestDetector().fit(make_training_frame())
    assert detector.feature_columns == ["a", "b"]


def test_extreme_row_gets_a_negative_score() -> None:
    detector = IsolationForestDetector(contamination=0.02)
    detector.fit(make_training_frame())
    extreme = pd.DataFrame({"a": [50.0], "b": [-50.0]})
    scores = detector.decision_scores(extreme)
    assert scores.iloc[0] < 0
    assert detector.is_atypical(scores).iloc[0]


def test_typical_rows_are_mostly_not_flagged() -> None:
    training = make_training_frame()
    detector = IsolationForestDetector().fit(training)
    flags = detector.is_atypical(detector.decision_scores(training))
    assert flags.mean() < 0.1


def test_decision_scores_keep_the_index() -> None:
    training = make_training_frame()
    detector = IsolationForestDetector().fit(training)
    rows = training.tail(5)
    assert detector.decision_scores(rows).index.equals(rows.index)


def test_decision_scores_reject_different_columns() -> None:
    detector = IsolationForestDetector().fit(make_training_frame())
    with pytest.raises(ValueError, match="feature columns differ"):
        detector.decision_scores(pd.DataFrame({"a": [1.0], "c": [2.0]}))


def test_save_and_load_give_the_same_scores(tmp_path: Path) -> None:
    training = make_training_frame()
    detector = IsolationForestDetector(random_seed=7).fit(training)
    path = tmp_path / "detector.joblib"
    detector.save(path)
    loaded = IsolationForestDetector.load(path)
    pd.testing.assert_series_equal(
        detector.decision_scores(training.head(20)),
        loaded.decision_scores(training.head(20)),
    )
