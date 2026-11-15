"""Walk-forward scores: each day is judged with a model fitted on the past."""

import numpy as np
import pandas as pd

from outlier_detector.models.isolation_forest_detector import (
    IsolationForestDetector,
)

TRAIN_WINDOW = 252
REFIT_EVERY = 21


def walk_forward_scores(
    features: pd.DataFrame,
    train_window: int = TRAIN_WINDOW,
    refit_every: int = REFIT_EVERY,
    contamination: float = 0.02,
    random_seed: int = 42,
) -> pd.Series:
    """Score every day using only the rows that came before it.

    A detector is fitted on the ``train_window`` rows before a block and
    then scores the next ``refit_every`` rows. The process repeats until
    the end of ``features``, so no score uses future data.

    Parameters
    ----------
    features : pd.DataFrame
        Feature rows sorted by date, without missing values.
    train_window : int, optional
        Number of rows in each training window. The default is 252, about
        one trading year.
    refit_every : int, optional
        Number of rows scored before the detector is refitted. The default
        is 21, about one trading month.
    contamination : float, optional
        Expected share of atypical rows, passed to the detector. The
        default is 0.02.
    random_seed : int, optional
        Seed of the detectors. The default is 42.

    Returns
    -------
    pd.Series
        Scores for the rows after the first ``train_window``, indexed like
        ``features``. Empty when there are not enough rows.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> features = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
    >>> walk_forward_scores(features, train_window=200).shape
    (100,)
    """
    scores = pd.Series(np.nan, index=features.index)
    for block_start in range(train_window, len(features), refit_every):
        block_end = min(block_start + refit_every, len(features))
        training_rows = features.iloc[block_start - train_window : block_start]
        block_rows = features.iloc[block_start:block_end]
        detector = IsolationForestDetector(
            contamination=contamination,
            random_seed=random_seed,
        ).fit(training_rows)
        scores.iloc[block_start:block_end] = detector.decision_scores(
            block_rows
        ).to_numpy()
    return scores.dropna()
