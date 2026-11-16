"""Walk-forward scores: each block is judged with a model fitted on the
past."""

import pandas as pd

from outlier_detector.models.isolation_forest_detector import (
    DEFAULT_CONTAMINATION,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_SEED,
    IsolationForestDetector,
)

TRAIN_MIN = 252 * 3
STEP = 126


def walk_forward_scores(
    features: pd.DataFrame,
    train_min: int = TRAIN_MIN,
    step: int = STEP,
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Score every day after ``train_min`` using only the days before it.

    The detector is refitted on all the rows before each block of ``step``
    days, and then scores that block, which it has not seen. The training
    window grows over time, so no score uses future data.

    Parameters
    ----------
    features : pd.DataFrame
        Feature rows sorted by date, without missing values.
    train_min : int, optional
        Number of rows in the first training window. The default is
        ``TRAIN_MIN``, about three trading years.
    step : int, optional
        Number of rows scored before the detector is refitted. The default
        is ``STEP``, about six months of trading days.
    contamination : float, optional
        Expected share of outliers. The default is 0.1.
    n_estimators : int, optional
        Number of isolation trees. The default is 300.
    random_seed : int, optional
        Seed of the detectors. The default is 42.

    Returns
    -------
    pd.DataFrame
        Scores of the rows from ``train_min`` onwards, with the columns
        ``score`` and ``is_atypical``. Empty when there are not enough rows.

    Example
    -------
    >>> walk_forward_scores(features, train_min=200, step=20)  # doctest: +SKIP
    """
    blocks = []
    for block_start in range(train_min, len(features), step):
        block_end = min(block_start + step, len(features))
        detector = IsolationForestDetector(
            contamination=contamination,
            n_estimators=n_estimators,
            random_seed=random_seed,
        ).fit(features.iloc[:block_start])
        blocks.append(
            detector.score_days(features.iloc[block_start:block_end])
        )
    if not blocks:
        return pd.DataFrame(columns=["score", "is_atypical"])
    return pd.concat(blocks)
