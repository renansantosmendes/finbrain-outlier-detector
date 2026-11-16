"""Tests of the walk-forward scores."""

import numpy as np
import pandas as pd

from outlier_detector.models.walk_forward import walk_forward_scores


def make_features(rows: int) -> pd.DataFrame:
    """Build normal feature rows.

    Parameters
    ----------
    rows : int
        Number of rows.

    Returns
    -------
    pd.DataFrame
        Frame with the columns ``a`` and ``b``.
    """
    generator = np.random.default_rng(1)
    return pd.DataFrame(generator.normal(size=(rows, 2)), columns=["a", "b"])


def test_scores_start_at_the_first_train_min_row() -> None:
    features = make_features(300)
    scores = walk_forward_scores(features, train_min=200, step=20)
    assert len(scores) == 100
    assert scores.index[0] == features.index[200]


def test_scores_are_empty_when_history_is_short() -> None:
    scores = walk_forward_scores(make_features(50), train_min=200)
    assert scores.empty
    assert list(scores.columns) == ["score", "is_atypical"]


def test_scores_cover_the_last_row() -> None:
    features = make_features(250)
    scores = walk_forward_scores(features, train_min=200, step=21)
    assert scores.index[-1] == features.index[-1]


def test_scores_do_not_depend_on_later_rows() -> None:
    features = make_features(300)
    full = walk_forward_scores(features, train_min=200, step=20)
    partial = walk_forward_scores(features.iloc[:240], train_min=200, step=20)
    pd.testing.assert_frame_equal(full.loc[partial.index], partial)
