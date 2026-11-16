"""Tests of the Isolation Forest detector."""

from pathlib import Path

import joblib
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


def test_extreme_row_gets_the_highest_score() -> None:
    training = make_training_frame()
    detector = IsolationForestDetector().fit(training)
    rows = pd.concat(
        [training.head(5), pd.DataFrame({"a": [50.0], "b": [-50.0]})],
        ignore_index=True,
    )
    scores = detector.score_days(rows)
    assert scores["score"].idxmax() == 5
    assert bool(scores.loc[5, "is_atypical"])


def test_score_days_returns_score_and_label_columns() -> None:
    training = make_training_frame()
    detector = IsolationForestDetector().fit(training)
    result = detector.score_days(training.tail(4))
    assert list(result.columns) == ["score", "is_atypical"]
    assert result.index.equals(training.tail(4).index)


def test_typical_rows_are_mostly_not_flagged() -> None:
    training = make_training_frame()
    detector = IsolationForestDetector(contamination=0.05).fit(training)
    flags = detector.score_days(training)["is_atypical"]
    assert flags.mean() < 0.2


def test_score_days_reject_different_columns() -> None:
    detector = IsolationForestDetector().fit(make_training_frame())
    with pytest.raises(ValueError, match="feature columns differ"):
        detector.score_days(pd.DataFrame({"a": [1.0], "c": [2.0]}))


def test_save_and_load_give_the_same_scores(tmp_path: Path) -> None:
    training = make_training_frame()
    detector = IsolationForestDetector(random_seed=7).fit(training)
    path = tmp_path / "detector.joblib"
    detector.save(path)
    loaded = IsolationForestDetector.load(path)
    pd.testing.assert_frame_equal(
        detector.score_days(training.head(20)),
        loaded.score_days(training.head(20)),
    )


def test_saved_file_is_a_plain_dictionary(tmp_path: Path) -> None:
    detector = IsolationForestDetector().fit(make_training_frame())
    path = tmp_path / "detector.joblib"
    detector.save(path)
    stored = joblib.load(path)
    assert isinstance(stored, dict)
    assert stored["feature_columns"] == ["a", "b"]
    assert set(stored) == {
        "forest",
        "feature_columns",
        "contamination",
        "n_estimators",
        "random_seed",
    }
