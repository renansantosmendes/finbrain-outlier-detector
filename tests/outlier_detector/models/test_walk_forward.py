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


def test_scores_start_after_the_first_window() -> None:
    features = make_features(300)
    scores = walk_forward_scores(features, train_window=200, refit_every=20)
    assert len(scores) == 100
    assert scores.index[0] == features.index[200]


def test_scores_are_empty_when_history_is_short() -> None:
    scores = walk_forward_scores(make_features(50), train_window=200)
    assert scores.empty


def test_scores_cover_the_last_row() -> None:
    features = make_features(250)
    scores = walk_forward_scores(features, train_window=200, refit_every=21)
    assert scores.index[-1] == features.index[-1]


def test_scores_do_not_depend_on_later_rows() -> None:
    features = make_features(300)
    full = walk_forward_scores(features, train_window=200, refit_every=20)
    partial = walk_forward_scores(
        features.iloc[:240],
        train_window=200,
        refit_every=20,
    )
    pd.testing.assert_series_equal(full.loc[partial.index], partial)
